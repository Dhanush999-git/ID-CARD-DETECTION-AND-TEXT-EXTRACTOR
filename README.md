### MINI PROJECT 
# steps to install and run:
1. install py -3.10
2. install and activate venv
3. pip3 install -r requirements-cpu.txt
   or
3. i) pip3 install tensorflow==2.17.1                                                                    
  ii) pip3 install -r requirements-modern.txt                                                                          
4. py -3.10 -m venv .venv .\.venv\Scripts\Activate.ps1
   
# Three Types to run:
-> python id_card_detection_image.py --image "image_path.jpg" --ocr --min_score 0.6                                                                                   
-> python id_card_detection_camera.py --min_score 0.5 --ocr                                                                                        
-> python app_flask.py --host 127.0.0.1 --port 5000 --min_score 0.5
