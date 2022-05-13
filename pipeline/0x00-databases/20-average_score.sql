-- calculates the average score for a student by creating a procedure
delimiter //
    CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INT)
    BEGIN
        SET @average = (SELECT AVG(corrections.score)
        FROM corrections
        WHERE corrections.user_id = user_id_new);
        UPDATE users SET average_score = @average WHERE id = user_id;
    END;
delimiter;