import cv2
import numpy as np
from fpdf import FPDF

# now generate a pdf where they have a width as passed in the function
import os

def aruco_to_pdf(id_list, marker_length_mm=19, total_length_mm=20, n_repeat=1, dicts=[('DICT_4X4_250', cv2.aruco.DICT_4X4_250)], only_return_imgs=False):
    assert marker_length_mm > 0, f'marker_length_mm must be positive'
    assert total_length_mm > 0, f'total_length_mm must be positive'
    assert marker_length_mm < total_length_mm, f'marker_length must be smaller than total_length_mm'
    aruco_imgs = []
    desired_tot_pix = 1000

    marker_px = int(desired_tot_pix * marker_length_mm / total_length_mm)
    pad_black = 1
    # pad_white = (desired_tot_pix - marker_px - 2*pad_black) // 2
    pad_white = (desired_tot_pix - marker_px) // 2
    print(f'Marker size in pixels: {marker_px} and white padding: {pad_white} and black padding: {pad_black}')
    # grey color
    # grey= (200, 200, 200)
    grey = (150, 150, 150)
    red = (0, 0, 255)
    black_pad_color = red



    for dict in dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict[1])
        for i, id in enumerate(id_list):
            img = cv2.aruco.generateImageMarker(aruco_dict, id, marker_px)
            # small white border
            img = cv2.copyMakeBorder(img, pad_white, pad_white, pad_white, pad_white, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # write the id in the image in small font
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2.4
            lineType = 4
            # cv2.putText(img, str(id), (0, 55), font, fontScale, grey, lineType)
            for i, sub_id in enumerate(str(id)):
                # cv2.putText(img, sub_id, (-2, 65*(i+1)), font, fontScale, grey, lineType)
                cv2.putText(img, sub_id, (-2, 65*i+img.shape[0]//2), font, fontScale, grey, lineType)
            
            # reshape img to have 3 channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if False:
                # small black border
                img = cv2.copyMakeBorder(img, pad_black, pad_black, pad_black, pad_black, cv2.BORDER_CONSTANT, value=grey)
            else: # black border on the outside such that image is increased by 2*pad_black
                color = (255, 255, 255)
                border_hori = np.ones((img.shape[0], pad_black, 3), dtype=np.uint8)*color
                img = np.concatenate((img, border_hori), axis=1)
                border_vert = np.ones((pad_black, img.shape[1], 3), dtype=np.uint8)*color
                img = np.concatenate((img, border_vert), axis=0)



            aruco_imgs.append(img)

    if only_return_imgs:
        return aruco_imgs
    # assert marker_px + 2*pad_black + 2*pad_white == desired_tot_pix, f'Ratio does not work out, tune marker_length_mm and total_length_mm'
    assert marker_px + 2*pad_white == desired_tot_pix, f'Ratio does not work out, tune marker_length_mm and total_length_mm'

    # must increase total_length_mm by the width of the black padding
    # we know that pad_white * 2 + marker_px converts to total_length_mm
    # so we can calculate the width of the black padding
    black_pad_mm = (pad_black / (pad_white * 2 + marker_px)) * total_length_mm
    print(f'Black padding in px: {pad_black} and in mm: {black_pad_mm}')
    total_length_with_black_mm = total_length_mm + black_pad_mm

    
    
    # now create A4 pdf with all aruco markers in it 
    page_size = (210, 297)
    orientation = 'L'
    if orientation == 'L':
        page_size = (page_size[1], page_size[0])
    # pad size with 10mm margin
    page_pad = 0
    repeat_pad = 0
    n_per_row = (page_size[0] - 2*page_pad) // total_length_with_black_mm
    max_rows = (page_size[1] - 2*page_pad - n_repeat*repeat_pad) // total_length_with_black_mm
    req_n_row = (len(id_list) + n_per_row - 1) // n_per_row
    assert req_n_row*n_repeat <= max_rows, f'Not enough space in the page to fit all markers'

    # create a pdf with all aruco markers
    dicts_str = "_".join([dict[0] for dict in dicts])
    total_length_mm_parsed = str(total_length_mm).replace('.', '_')
    pdf_path = f'markers_{"_".join([str(id) for id in id_list[:10]])}_dicts_{dicts_str}_totalL_{total_length_mm_parsed}mm.pdf'
    # create new a5 pdf
    pdf = FPDF(orientation=orientation, unit='mm', format='A4') # orientations: P=portrait, L=landscape
    pdf.set_auto_page_break(auto=True, margin=0)
    pdf.add_page()

    # pdf.set_font("Arial", size=12)
    # # write title with the ids
    # pdf.set_xy(5, 5)
    # pdf.cell(0, 8, f'IDs: {id_list[0]} - {id_list[-1]}, dict: {dict[0]}, marker L: {marker_length_mm}mm, total L: {total_length_mm}mm', 0, 1, 'C')
    

    drawn_rows = {}
    drawn_cols = {}
    for i, id in enumerate(id_list):
        x_offset = page_pad + (i % n_per_row) * total_length_with_black_mm
        curr_row = i // n_per_row
        curr_col = i % n_per_row
        # pdf.image(img_path, x_offset, y_offset, w=total_length_mm, h=total_length_mm)
        if len(dicts) == 1:
            img = aruco_imgs[i]
            cv2.imwrite(f'aruco_{id}.png', img)
            img_path = f'aruco_{id}.png'
        for j in range(n_repeat):
            if len(dicts) > 1:
                img = aruco_imgs[i + j*len(id_list)]
                img_path = f'{dicts[j][0]}_aruco_{id}.png'
                cv2.imwrite(img_path, img)
            y_offset = (j+1)*repeat_pad
            y_offset += page_pad + (i // n_per_row) * total_length_with_black_mm + req_n_row*total_length_with_black_mm*j
            pdf.image(img_path, x_offset, y_offset, w=total_length_with_black_mm, h=total_length_with_black_mm)
            # create black vertical padding before img if curr_col == 0 
            black_pad_color_rgb = (black_pad_color[2], black_pad_color[1], black_pad_color[0])
            if False:
                if curr_col == 0:
                    pdf.set_fill_color(*black_pad_color_rgb)
                    pdf.rect(x_offset-black_pad_mm, y_offset-black_pad_mm, black_pad_mm, total_length_with_black_mm+black_pad_mm, 'F')
                # create black horizontal padding before img if curr_row == 0
                if curr_row == 0:
                    pdf.set_fill_color(*black_pad_color_rgb)
                    pdf.rect(x_offset, y_offset-black_pad_mm, total_length_with_black_mm, black_pad_mm, 'F')
            else:
                # draw for each col a vertical cutting line
                if curr_col not in drawn_cols:
                    drawn_cols[curr_col] = x_offset-black_pad_mm
                    # if it is last column, append also another one at the end
                    if curr_col == n_per_row - 1:
                        drawn_cols[curr_col+1] = x_offset + total_length_with_black_mm

                # draw for each row a horizontal cutting line
                if curr_row + (j+2)*req_n_row not in drawn_rows:
                    drawn_rows[curr_row + (j+2)*req_n_row] = y_offset-black_pad_mm
                    # if it is last row, append also another one at the end
                    if curr_row == req_n_row - 1:
                        drawn_rows[curr_row + (j+2)*req_n_row+1] = y_offset + total_length_with_black_mm

            if len(dicts) > 1:
                os.remove(img_path)
        if len(dicts) == 1:
            os.remove(img_path)

    # write title at bottom instead
    pdf.set_font("Arial", size=12)
    pdf.set_xy(5, page_size[1] - 7)
    pdf.cell(0, 7, f'IDs: {id_list[0]} - {id_list[-1]}, dicts: {", ".join([dict[0] for dict in dicts])}, marker L: {marker_length_mm}mm, total L: {total_length_mm}mm', 0, 1, 'C')

    # draw cutting lines
    for x_offset in drawn_cols.values():
        pdf.set_fill_color(*black_pad_color_rgb)
        pdf.rect(x_offset, 0, black_pad_mm, page_size[1], 'F')

    for y_offset in drawn_rows.values():
        pdf.set_fill_color(*black_pad_color_rgb)
        pdf.rect(0, y_offset, page_size[0], black_pad_mm, 'F')

        


    # pdf.output(pdf_path)
    # print(f'PDF created at {pdf_path}')
    pdf.output('temp.pdf')


first_id = 0
n_markers = 6*11
id_range = range(first_id, first_id + n_markers)
id_list = [i for i in id_range]
dicts = []
dicts.append(('DICT_4X4_250', cv2.aruco.DICT_4X4_250))
aruco_to_pdf(id_list, marker_length_mm=20, total_length_mm=23, n_repeat=2, dicts=dicts, only_return_imgs=False)