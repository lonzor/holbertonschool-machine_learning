-- lists bands with glam rock listed as main style
SELECT band_name, IF(split is NULL, (2020 - formed), (split - formed)) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC