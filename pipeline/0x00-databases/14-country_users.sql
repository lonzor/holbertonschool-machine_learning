-- creates a table "users"
CREATE TABLE IF NOT EXISTS users (
	id INT NOT ALL AUTO_INCREMENT,
	email VARCHAR(255) NOT NULL UNIQUE,
	name VARCHAR(256),
	country ENUM('US', 'CO', 'TN') NOT NULL,
	PRIMARY KEY (id));
