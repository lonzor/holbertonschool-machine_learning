-- creates a trigger to add a bonus for a students grade
delimiter //
CREATE PROCEDURE AddBonus(
    IN user_id_new INT,
    IN project_name varchar(255),
    IN score_name INT)
    BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
            INSERT INTO projects(name) VALUES (project_name);
        END IF;
        INSERT INTO corrections(user_id, project_id, score)
            VALUES(user_id_new, (SELECT id FROM projects WHERE name = project_name), score_new);
    END //
delimiter;