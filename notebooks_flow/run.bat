cd C:\git\cuponesWong\CuponesWong\notebooks_flow

call activate audiopro
python getting_areas.py

call activate tfkeras
cd yolo
python ocr_local_reco.py
cd ..

call activate tfkeras
python azure_process.py
