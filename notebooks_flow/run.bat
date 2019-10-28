cd C:\git\cuponesWong\CuponesWong\notebooks_flow

call activate audiopro
python getting_areas.py -c 4

call activate tfkeras
cd yolo
python ocr_local_reco.py -c 4
cd ..

call activate tfkeras
python azure_process.py -c 4
