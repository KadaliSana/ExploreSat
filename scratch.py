import numpy as np

def test(h, w, tile_size, overlap):
    stride = tile_size - overlap
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    
    if (h + pad_h - tile_size) % stride != 0:
        pad_h += stride - ((h + pad_h - tile_size) % stride)
    if (w + pad_w - tile_size) % stride != 0:
        pad_w += stride - ((w + pad_w - tile_size) % stride)
        
    ph = h + pad_h
    pw = w + pad_w
    
    print(f"h={h}, ph={ph}, max_y = {ph - tile_size + 1}")
    last_y = -1
    for y in range(0, ph - tile_size + 1, stride):
        last_y = y
    
    print(f"last_y={last_y}, last_y+tile_size={last_y+tile_size}, Covers h? {last_y+tile_size >= h}")

test(4581, 5703, 512, 64)
test(100, 200, 512, 64)
