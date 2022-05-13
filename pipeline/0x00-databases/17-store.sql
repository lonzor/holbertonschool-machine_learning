-- creates trigger that decreases quantity of an item when it is purchased
CREATE TRIGGER decrease_items
AFTER INSERT
ON orders
FOR EACH ROW
    UPDATE decrease_items
    SET quantity = quantity - NEW.number
    WHERE items.name = NEW.item_name;