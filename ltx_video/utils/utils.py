def calculate_new_dimensions(canvas_height, canvas_width, height, width, fit_into_canvas, block_size = 16):
    if fit_into_canvas:
        scale1  = min(canvas_height / height, canvas_width / width)
        scale2  = min(canvas_width / height, canvas_height / width)
        scale = max(scale1, scale2) 
    else:
        scale = (canvas_height * canvas_width / (height * width))**(1/2)

    new_height = round( height * scale / block_size) * block_size
    new_width = round( width * scale / block_size) * block_size
    return new_height, new_width