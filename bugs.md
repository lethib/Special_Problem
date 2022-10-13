### Bugs

- Sample not cleared every time it has been completed (Done)
- doesn't skip `height` lines (Done)

Pixel scrapping + delta values and difference mean values computation ok !

- Position is not correct (Done)
- Force the x axes value to fit the image width

Results are not good :

We can use RANSAC (have to talk with M.Pradalier to put that in place) -> Requires the equation of the interface -> can be approximate using geogebra or ImageJ

Then for RANSAC we have to form a DataSet (i.e. a PointCloud) -> all dark pixels can be transformed into points