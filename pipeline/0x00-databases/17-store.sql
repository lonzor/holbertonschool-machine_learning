-- creates trigger that decreases quantity of an item when it is purchased
delimiter //
CREATE TRIGGER decrease_items
    AFTER INSERT ON orders FOR EACH ROW
BEGIN
    update items SET quantity = quantity - new.number WHERE items.name=new.item_name;
END //
delimiter;