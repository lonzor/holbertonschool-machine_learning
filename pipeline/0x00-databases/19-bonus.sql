-- creates a trigger to add a bonus for a students grade
delimiter //
CREATE PROCEDURE AddBonus(user_id_new INT, project_name varchar(255), score_new FLOAT)
    BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
            INSERT INTO projects(name) VALUES (project_name);
        END IF;
        INSERT INTO corrections(user_id, project_id, score)
            VALUES(user_id_new, (SELECT id FROM projects WHERE name = project_name), score_new);
    END //
delimiter;