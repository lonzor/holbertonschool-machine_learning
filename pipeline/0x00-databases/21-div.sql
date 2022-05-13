-- creates function that divides the first number by second
delimiter //
    CREATE FUNCTION SafeDiv (a INT, b INT)
    RETURNS FLOAT
    BEGIN
        IF b = 0 THEN RETURN 0;
        END IF;
        RETURN (a / b);
    END //
delimiter;