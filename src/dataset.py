import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple
import psycopg2

class SteamAppsDataset(Dataset):
    def __init__(self, db_params):
        super().__init__()

        self.apps: pd.DataFrame
        self.tags: pd.DataFrame
        self.tag_id2label: Dict[int, str]
        self.tag_label2id: Dict[str, int]

        conn = psycopg2.connect(**db_params) # type: ignore
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT app_id, title, description FROM game ORDER BY app_id")
                self.apps = pd.DataFrame(cur.fetchall(), columns=["app_id", "title", "description"])

                cur.execute("SELECT app_id, tag_id FROM game_tag")
                self.tags = pd.DataFrame(cur.fetchall(), columns=["app_id", "tag_id"])

                cur.execute("SELECT tag_id, \"name\" FROM tag")
                result = cur.fetchall()
                self.tag_id2label = {id: name for id, name in result}
                self.tag_label2id = {name: id for id, name in result}
        
        self.all_tag_ids = sorted(self.tag_id2label.keys()) # сортируем для детерминированности

        one_hot_df = pd.crosstab(
            self.tags['app_id'],
            self.tags['tag_id'],
            dropna=False
        ).clip(upper=1)

        one_hot_df = one_hot_df.reindex(columns=self.all_tag_ids, fill_value=0)
        one_hot_df = one_hot_df.reindex(self.apps['app_id'], fill_value=0)
        self.tags_one_hot = torch.tensor(one_hot_df.values)
    
    def __getitem__(self, index) -> Tuple[str, torch.Tensor]:
        app_row = self.apps.loc[index]
        description = app_row["description"]
        target_vector = self.tags_one_hot[index]
        return (description, target_vector)
    
    def __len__(self) -> int:
        return len(self.apps)
