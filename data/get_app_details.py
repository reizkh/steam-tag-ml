import pandas as pd
import psycopg2
from typing import Self, List, Dict
import requests
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from bs4 import BeautifulSoup
import omegaconf


LOG: logging.Logger


class DataBuffer:
    def __init__(self, db_params, buffer_size: int = 1000):
        self.buffer = []
        self.db_params = db_params
        self.buffer_size = buffer_size
        self.pbar = tqdm(total=self.buffer_size, leave=False, desc="Buffer progress")
    
    def __enter__(self) -> Self:
        self.conn: psycopg2.extensions.connection = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.buffer:
            self.process()
        self.cursor.close()
        self.conn.close()
    
    def insert(self, id_list: pd.Series) -> None:
        unique_ids = id_list.unique()
        self.buffer.extend(unique_ids.tolist())
        
        self.pbar.n = min(len(self.buffer), self.buffer_size)
        self.pbar.update()
        
        if len(self.buffer) >= self.buffer_size:
            self.process()
    
    def process(self) -> None:
        self.pbar.close()
        filtered = self.filter_new(self.buffer)
        LOG.info(f"Skipping {len(self.buffer)-len(filtered)} duplicates")
        
        processing_pbar = tqdm(filtered, desc="Processing batch", leave=False)
        with self.conn:
            for app_id in processing_pbar:
                try:
                    data = self.fetch_details(app_id)
                    name, tags, desc = data.get("name", ""), data.get("tags", []), data.get("description", "")

                    self.insert_app(app_id, name, desc)
                    if tags:
                        query = """
                            WITH new_tags AS (
                                INSERT INTO tag (name)
                                VALUES ( unnest( %(tag_list)s::text[] ) )
                                ON CONFLICT DO NOTHING
                            )
                            SELECT tag_id FROM tag
                            WHERE name = ANY( %(tag_list)s::text[] );
                        """
                        self.cursor.execute(query, {"tag_list": tags})
                        result = self.cursor.fetchall()

                        tag_id_list = [x[0] for x in result]
                        self.insert_tags(app_id, tag_id_list)

                    processing_pbar.set_postfix(app_id=app_id, tags_found=len(tags))
                except Exception as e:
                    LOG.error(f"Error while processing app_id {app_id}: {str(e)}")
        self.buffer = []
        self.pbar = tqdm(total=self.buffer_size, leave=False, desc="Buffer progress")
    
    @staticmethod
    def fetch_details(app_id: int) -> Dict:
        endpoint = f"https://store.steampowered.com/app/{app_id}"
        r = requests.get(endpoint, timeout=10)
        
        if r.status_code != 200:
            raise LookupError(f"Failed to fetch data for app_id {app_id}")
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        try:
            desc = soup.select(".game_description_snippet")[0].get_text().strip()
            name = soup.select("#appHubAppName")[0].get_text().strip()
            tags = [x.get_text().strip() for x in soup.select(".app_tag")]
        except Exception:
            raise LookupError(f"Data for app_id {app_id} not found")

        return {"name": name, "tags": tags, "description": desc}
    
    def filter_new(self, app_id_list: List[int]) -> List[int]:
        query = """
            SELECT unnest( %(app_id_list)s::int[] )
            EXCEPT
            SELECT app_id FROM game
        """
        self.cursor.execute(query, {"app_id_list": app_id_list})
        result = self.cursor.fetchall()
        return [x[0] for x in result]

    def insert_app(self, app_id: int, name: str, desc: str) -> None:
        query = """
            INSERT INTO game ( app_id, title, description )
            VALUES ( %(app_id)s, %(name)s, %(desc)s )
            ON CONFLICT DO NOTHING
        """
        self.cursor.execute(query, {"app_id": app_id, "name": name, "desc": desc})

    def insert_tags(self, app_id: int, tag_id_list: List[int]) -> None:
        query = """
            INSERT INTO game_tag ( app_id, tag_id )
            VALUES ( %(app_id)s, unnest( %(tag_id_list)s::int[] ) )
            ON CONFLICT DO NOTHING
        """
        self.cursor.execute(query, {"app_id": app_id, "tag_id_list": tag_id_list})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr,
        force=True
    )
    
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.DEBUG)

    filename = "data/raw/dataset.csv"
    data = pd.read_csv(filename, chunksize=100_000) # 6417106 lines
    db_params = omegaconf.OmegaConf.load("configs/db_params.yaml")

    with DataBuffer(db_params, buffer_size=1000) as buffer:
        with logging_redirect_tqdm():
            pbar = tqdm(data, leave=False, total=6_417_106//100_000+1, desc=f"Reading {filename}")
            for df in pbar:
                ids: pd.Series = df['app_id']
                buffer.insert(ids)
