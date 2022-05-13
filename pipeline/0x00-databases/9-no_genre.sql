-- lists all tv shows without a genre
SELECT tv_shows.title, tv_show_genres.genre_id AS genre_id FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id is NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
