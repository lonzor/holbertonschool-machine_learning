-- calculates the average score for a student by creating a procedure
delimiter //
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INT)
    BEGIN
        UPDATE users
        SET average_score = (
            SELECT AVG(score)
            FROM corrections WHERE corrections.user_id = user_id_new)
            WHERE id = user_id;
    END;
delimiter;