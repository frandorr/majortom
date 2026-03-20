import numpy as np
import shapely.geometry
from shapely.geometry import Polygon
from geolib import geohash


class GridCell:

    def __init__(self, geom: shapely.geometry.Polygon, is_primary: bool = True):
        self.geom = geom
        self.is_primary = is_primary
        self._id = geohash.encode(geom.centroid.y, geom.centroid.x, 11)

    def id(self) -> str:
        return self._id


class MajorTomGrid:
    def __init__(self, d: int = 320, overlap=True):
        if d <= 0:
            raise ValueError("Grid spacing must be positive")
        self.D = d
        self.earth_radius = 6378137
        self.overlap = overlap
        self.row_count = max(2, np.ceil(np.pi * self.earth_radius / self.D))
        self.lat_spacing = self.get_lat_spacing()
        self._lat_offset = self.lat_spacing / 2 if int(self.row_count) % 2 else 0

    def get_lat_spacing(self):
        return min(180 / self.row_count, 89)

    def get_row_lat(self, row_idx):
        return -90 + self._lat_offset + row_idx * self.lat_spacing

    def get_lon_spacing(self, lat):
        lat_rad = np.radians(min(max(lat, -89), 89))
        circumference = 2 * np.pi * self.earth_radius * np.cos(lat_rad)
        n_cols = int(np.ceil(circumference / self.D))
        return 360 / max(n_cols, 1)

    def get_lon_offset(self, lon_spacing):
        n_cols = round(360 / lon_spacing) if lon_spacing > 0 else 0
        return lon_spacing / 2 if n_cols % 2 else 0

    def get_col_lon(self, col_idx, lon_spacing, lon_offset):
        return -180 + lon_offset + col_idx * lon_spacing

    def generate_grid_cells(self, polygon):
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        if min_lon > max_lon:
            max_lon += 360

        start_row = int(np.floor((min_lat + 90 - self._lat_offset) / self.lat_spacing))
        end_row = int(np.ceil((max_lat + 90 - self._lat_offset) / self.lat_spacing))

        while self.get_row_lat(start_row) > min_lat + 1e-10:
            start_row -= 1
        while self.get_row_lat(end_row) < max_lat - 1e-10:
            end_row += 1

        for row_idx in range(start_row, end_row + 1):
            lat = self.get_row_lat(row_idx)
            lon_spacing = self.get_lon_spacing(lat)
            lon_offset = self.get_lon_offset(lon_spacing)

            start_col = int(np.floor((min_lon + 180 - lon_offset) / lon_spacing))
            end_col = int(np.ceil((max_lon + 180 - lon_offset) / lon_spacing))

            while self.get_col_lon(start_col, lon_spacing, lon_offset) > min_lon + 1e-10:
                start_col -= 1
            while self.get_col_lon(end_col, lon_spacing, lon_offset) < max_lon - 1e-10:
                end_col += 1

            for col_idx in range(start_col, end_col + 1):
                lon = self.get_col_lon(col_idx, lon_spacing, lon_offset)
                primary_cell_polygon = Polygon([
                    [lon, lat],
                    [lon + lon_spacing, lat],
                    [lon + lon_spacing, lat + self.lat_spacing],
                    [lon, lat + self.lat_spacing]
                ])

                if self.overlap:
                    overlap_lon = lon + lon_spacing/2
                    overlap_lat = lat + self.lat_spacing/2
                    overlap_cell_polygon = Polygon([
                        [overlap_lon, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat + self.lat_spacing],
                        [overlap_lon, overlap_lat + self.lat_spacing]
                    ])

                    if primary_cell_polygon.intersects(polygon):
                        yield GridCell(primary_cell_polygon, is_primary=True)
                    if overlap_cell_polygon.intersects(polygon):
                        yield GridCell(overlap_cell_polygon, is_primary=False)
                else:
                    if primary_cell_polygon.intersects(polygon):
                        yield GridCell(primary_cell_polygon, is_primary=True)


    def cell_from_id(self, cell_id: str) -> GridCell:
        search_id = cell_id[:11] if len(cell_id) > 11 else cell_id
        if len(search_id) != 11:
            raise ValueError("Cell ID must be at least 11 characters")

        bounds = geohash.bounds(search_id)
        center_lat = (bounds.sw[0] + bounds.ne[0]) / 2
        center_lon = (bounds.sw[1] + bounds.ne[1]) / 2

        half_lat = self.lat_spacing / 2
        for row_offset in (-1, 0, 1):
            row_idx = int(np.floor((center_lat + 90 - self._lat_offset) / self.lat_spacing)) + row_offset
            row_lat = self.get_row_lat(row_idx)
            lon_spacing = self.get_lon_spacing(row_lat)
            lon_offset = self.get_lon_offset(lon_spacing)
            half_lon = lon_spacing / 2

            for col_offset in (-1, 0, 1):
                col_idx = int(np.floor((center_lon + 180 - lon_offset) / lon_spacing)) + col_offset
                cell_lon = self.get_col_lon(col_idx, lon_spacing, lon_offset)

                primary = Polygon([
                    [cell_lon, row_lat],
                    [cell_lon + lon_spacing, row_lat],
                    [cell_lon + lon_spacing, row_lat + self.lat_spacing],
                    [cell_lon, row_lat + self.lat_spacing],
                ])
                candidate = GridCell(primary, is_primary=True)
                if candidate.id() == search_id:
                    return candidate

                if self.overlap:
                    overlap_lon = cell_lon + half_lon
                    overlap_lat = row_lat + half_lat
                    overlap_poly = Polygon([
                        [overlap_lon, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat],
                        [overlap_lon + lon_spacing, overlap_lat + self.lat_spacing],
                        [overlap_lon, overlap_lat + self.lat_spacing],
                    ])
                    overlap_cell = GridCell(overlap_poly, is_primary=False)
                    if overlap_cell.id() == search_id:
                        return overlap_cell

        raise ValueError(f"No cell found with ID {cell_id}")

    def migrate_cell_id(self, old_id: str) -> GridCell:
        """Map a cell ID from a prior grid version to the current grid.

        Decodes the geohash to recover the approximate centroid, then returns
        the current-grid cell that contains that point.
        """
        search_id = old_id[:11] if len(old_id) > 11 else old_id
        if len(search_id) != 11:
            raise ValueError("Cell ID must be at least 11 characters")

        lat, lon = (float(v) for v in geohash.decode(search_id))

        row_idx = int(np.floor((lat + 90 - self._lat_offset) / self.lat_spacing))
        row_lat = self.get_row_lat(row_idx)
        lon_spacing = self.get_lon_spacing(row_lat)
        lon_offset = self.get_lon_offset(lon_spacing)
        col_idx = int(np.floor((lon + 180 - lon_offset) / lon_spacing))
        cell_lon = self.get_col_lon(col_idx, lon_spacing, lon_offset)

        cell_polygon = Polygon([
            [cell_lon, row_lat],
            [cell_lon + lon_spacing, row_lat],
            [cell_lon + lon_spacing, row_lat + self.lat_spacing],
            [cell_lon, row_lat + self.lat_spacing],
        ])
        return GridCell(cell_polygon, is_primary=True)