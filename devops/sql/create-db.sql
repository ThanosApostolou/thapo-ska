CREATE DATABASE thapo_ska_db_local WITH ENCODING = 'UTF8';
-- Change connection to thapo_ska_db_local database
CREATE USER example_user WITH ENCRYPTED PASSWORD 'example_password';
CREATE SCHEMA IF NOT EXISTS thapo_ska_schema_local;
GRANT ALL PRIVILEGES ON DATABASE thapo_ska_db_local to example_user;
GRANT ALL PRIVILEGES ON SCHEMA thapo_ska_schema_local to example_user;

CREATE DATABASE thapo_skakeycloak_db_local WITH thapo_skakeycloak_db_local = 'UTF8';
-- Change connection to thapo_skakeycloak_db_local database
CREATE USER example_keycloak_user WITH ENCRYPTED PASSWORD 'example_keycloak_password';
CREATE SCHEMA IF NOT EXISTS thapo_skakeycloak_schema_local;
GRANT ALL PRIVILEGES ON DATABASE thapo_skakeycloak_db_local to example_keycloak_user;
GRANT ALL PRIVILEGES ON SCHEMA thapo_skakeycloak_schema_local to example_keycloak_user;
