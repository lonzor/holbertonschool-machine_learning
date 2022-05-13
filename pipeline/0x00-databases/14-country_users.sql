-- creates a table "users"
DROP TABLE IF EXISTS 'users';
CREATE TABLE users (
	id INT NOT ALL AUTO_INCREMENT,
	email VARCHAR(255) UNIQUE NOT ALL,
	name VARCHAR(256),
	country ENUM('US', 'CO', 'TN') DEFAULT 'US' NOT NULL,
	PRIMARY KEY (id));
