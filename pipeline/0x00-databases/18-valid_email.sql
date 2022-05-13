-- resets valid_email when the email has been changed
CREATE TRIGGER
BEFORE UPDATE
ON users
FOR EACH ROW
    if STRCMP(old.email, new.email) <> 0 THEN
        SET new.valid_email = 0;
    END IF;