-- DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION pg_database_owner;

-- DROP SEQUENCE review_id_seq;

CREATE SEQUENCE review_id_seq
	MINVALUE 0
	NO MAXVALUE
	START 0
	NO CYCLE;
-- DROP SEQUENCE tag_tag_id_seq;

CREATE SEQUENCE tag_tag_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;-- public.game definition

-- Drop table

-- DROP TABLE game;

CREATE TABLE game (
	app_id int4 NOT NULL,
	title text NOT NULL,
	description text NULL,
	CONSTRAINT game_pk PRIMARY KEY (app_id)
);


-- public.tag definition

-- Drop table

-- DROP TABLE tag;

CREATE TABLE tag (
	"name" text NOT NULL,
	tag_id serial4 NOT NULL,
	CONSTRAINT tag_pk PRIMARY KEY (tag_id),
	CONSTRAINT tag_unique UNIQUE (name)
);


-- public.game_tag definition

-- Drop table

-- DROP TABLE game_tag;

CREATE TABLE game_tag (
	app_id int4 NOT NULL,
	tag_id int4 NOT NULL,
	CONSTRAINT game_tag_unique UNIQUE (app_id, tag_id),
	CONSTRAINT game_tag_game_fk FOREIGN KEY (app_id) REFERENCES game(app_id) ON DELETE CASCADE,
	CONSTRAINT game_tag_tag_fk FOREIGN KEY (tag_id) REFERENCES tag(tag_id)
);