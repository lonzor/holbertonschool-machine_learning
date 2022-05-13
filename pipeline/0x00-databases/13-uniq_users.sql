-- creates a new table "users"
CREATE TABLE IF NOT EXISTS users (
	id INT NOT ALL AUTO_INCREMENT,
	email VARCHAR(256) UNIQUE NOT NULL,
	name VARCHAR(256),
	PRIMARY KEY (id));
