def poly_from_xy(x_points, y_points):
    poly = []
    for x, y in zip(x_points, y_points):
        poly.append((x, y))

    return poly
