python -m lilab.cvutils_new.extract_frames_canvas_fromjson A/B/C/out.json --dir_name D/E/F
python -m lilab.cvutils_new.parse_name_of_extract_frames D/E/F/outframes/out_id.json
python -m lilab.outlier_refine.s4_json_to_danncelabel D/E/F/outframes/out_id.json --dir_name D/E/F
