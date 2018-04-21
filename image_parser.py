import os
from nba_card_parser import preprocess_predict, get_format, fix_ratio_if_needed, calc_height_rect, calc_stat_rect, get_height, get_stats, get_ovr, get_name, get_pos, get_type, FORMAT1, FORMAT2

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# API_KEY = "8f4626b39f88957"
# API_URL = "https://api.ocr.space/parse/image"

from sklearn.externals import joblib
clf, pp = joblib.load('Training/PKL/adv_stats_digits.pkl')
height_clf, height_pp = joblib.load('Training/PKL/height_digits.pkl')
ovr_clf, ovr_pp = joblib.load('Training/PKL/ovr_digits.pkl')
pos_clf, pos_pp = joblib.load('Training/PKL/pos.pkl')
type_clf, type_pp = joblib.load('Training/PKL/type.pkl')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = os.path.join(BASE_DIR, "raw_cards")
folder = os.path.join(folder, "new_raw_cards")
# folder = "C:\Users\hkuang\Pictures\MEmu Photo\Screenshots"

for f in os.listdir(folder):
  image_path = os.path.join(folder, f)
  if os.path.isdir(image_path):
    continue
  img = Image.open(image_path)
  img = fix_ratio_if_needed(img)
  card_format = get_format(img)
  #print get_stats(img, clf, pp, card_format, save=True, save_path='Training/adv_stats_digits')
  #height = get_height(img, height_clf, height_pp, card_format, save=True, save_path='Training/height_digits')
  #print height
  print get_ovr(img, ovr_clf, ovr_pp, card_format, save=True, save_path="Training/ovr_digits")

  # print get_name(img, height, card_format, save=True)
  #print get_pos(img, pos_clf, pos_pp, card_format, save=True, save_path='Training/pos')
  #print get_type(img, type_clf, type_pp, card_format, save=True, save_path='Training/type')

# image_path = "C:\Users\hkuang\Desktop\\nba-card-parser\\new_raw_cards\\29066880_2175510069343365_6333750687004840297_n.jpg"
# img = Image.open(image_path)
# card_format = get_format(img)
# # get_stats(img, clf, pp, card_format, save=True, save_path='Training/adv_stats_digits')
# # get_height(img, height_clf, height_pp, card_format, save=True, save_path='Training/height_digits')
# print get_ovr(img, ovr_clf, ovr_pp, card_format, save=True, save_path="Training/ovr_digits")
