-- resets valid_email when the email has been changed
delimiter //
CREATE TRIGGER change_email
        BEFORE UPDATE
        ON users
        FOR EACH ROW
        BEGIN
            IF STRCMP(old.email, new.email) <> 0 THEN
                SET new.valid_email = 0;
            END IF;
        END //
delimiter;