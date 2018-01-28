import PIL
from PIL import Image, ImageFilter
import os
import base64
import numpy as np
import cv2
import json
import hashlib
import re
import regex
from skimage.feature import hog
import io
import pytesseract
import threadpool
import pymongo
from pymongo import MongoClient
from cStringIO import StringIO
from optparse import OptionParser
from sklearn.externals import joblib

import logger
logconfig = logger.LoggerConfigurator()
parse_logger = logconfig.get_logger(__name__)

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UNIVERSAL = "universal"
FORMAT1 = "format1"
FORMAT2 = "format2"

stats_format_xy = {
  UNIVERSAL: {
    "DIGIT_WIDTH_RATIO": 11.0/1090.0,
    "DIGIT_HEIGHT_RATIO": 25.0/613.0,
    "ROW_HEIGHT_RATIO": 30.4/613.0,
    "COL_DIFF_RATIO": 280/1090.0,
  },
  FORMAT1: {
    "STATS_START_X_RATIO": 575.0/1090.0,
    "STATS_START_Y_RATIO": 158.0/613.0,
  },
  FORMAT2: {
    "STATS_START_X_RATIO": 900.0/1706.0,
    "STATS_START_Y_RATIO": 285.0/960.0,
  },
}

name_format_xy = {
  UNIVERSAL: {
    "NAME_WIDTH_RATIO": 170.0/1707.0,
    "NAME_HEIGHT_RATIO": 86.0/960.0,
  },
  FORMAT1: {
    "NAME_START_X_RATIO": 320.0/1707.0,
    "NAME_START_Y_RATIO": 384.0/960.0,
  },
  FORMAT2: {
    "NAME_START_X_RATIO": 318/1706.0,
    "NAME_START_Y_RATIO": 425/960.0,
  }
}

def calc_stat_rect(column, row, digit, width, height, card_format):
  row_height = stats_format_xy[UNIVERSAL]["ROW_HEIGHT_RATIO"] * height
  col_diff = stats_format_xy[UNIVERSAL]["COL_DIFF_RATIO"] * width
  digit_width = stats_format_xy[UNIVERSAL]["DIGIT_WIDTH_RATIO"] * width
  digit_height = stats_format_xy[UNIVERSAL]["DIGIT_HEIGHT_RATIO"] * height
  start_x = stats_format_xy[card_format]["STATS_START_X_RATIO"] * width
  start_y = stats_format_xy[card_format]["STATS_START_Y_RATIO"] * height
  x1 = start_x + (col_diff * (column - 1)) + (digit_width * (digit - 1))
  y1 = start_y + (row_height * (row - 1))
  x2 = x1 + digit_width
  y2 = y1 + digit_height
  return (x1, y1, x2, y2)

def calc_name_rect(width, height, start_y_adjust, card_format):
  start_x = name_format_xy[card_format]["NAME_START_X_RATIO"] * width
  start_y = name_format_xy[card_format]["NAME_START_Y_RATIO"] * height
  name_width = name_format_xy[UNIVERSAL]["NAME_WIDTH_RATIO"] * width
  name_height = name_format_xy[UNIVERSAL]["NAME_HEIGHT_RATIO"] * height
  x1 = start_x
  y1 = start_y + (start_y_adjust/960.0 * height)
  x2 = x1 + name_width
  y2 = y1 + name_height
  return (x1, y1, x2, y2)

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def get_first_last_name(player_name_guess):
  player_name_split = player_name_guess.split()
  if player_name_guess.startswith(" "):
    first_name_guess = ""
    last_name_guess = player_name_split[0]
  else:
    first_name_guess = player_name_split[0]
    #if len(player_name_split) > 1 and len(player_name_split[1]) > 1:
    last_name_guess = player_name_split[1]
    #else:
    #  last_name_guess = ""
  return first_name_guess, last_name_guess

def get_name_regex(player_name_guess):
  return player_name_guess.replace(' ', '.*\s.*')

def get_fuzzy_count(str1, str2, starting_num):
  for x in range(starting_num, 0, -1):
    matches = regex.search('(%s){e<%s}' % (str1, x), str2)
    if matches:
      return get_fuzzy_count(str1, str2, x-1)
    return x
  return starting_num

def find_closest_nba_player_name(player_name_guess, player_height, all_players_dict):
  lowest_fuzzy_count = 999
  lowest_fuzzy_player = None
  lowest_fuzzy_player_name_length_diff = 999
  for player_name in all_players_dict.keys():
    if not (player_height-2 <= all_players_dict[player_name]["height"] <= player_height+2):
      continue
    # first_name_guess, last_name_guess = get_first_last_name(player_name_guess)
    # regex = '%s.*\s.*%s' % (first_name_guess, last_name_guess)
    regex = get_name_regex(player_name_guess)
    fuzzy_count = get_fuzzy_count(regex, player_name, 5)
    player_first_name_length_diff = abs(len(player_name.split()[0]) - len(player_name_guess.split()[0]))
    if fuzzy_count < lowest_fuzzy_count or (fuzzy_count == lowest_fuzzy_count and player_first_name_length_diff < lowest_fuzzy_player_name_length_diff):
      lowest_fuzzy_count = fuzzy_count
      lowest_fuzzy_player = player_name
      lowest_fuzzy_player_name_length_diff = player_first_name_length_diff
      # print "lowest fuzzy count: %s || lowest_fuzzy_player: %s || length_diff: %s" % (lowest_fuzzy_count, lowest_fuzzy_player, player_first_name_length_diff)
  # fuzzy_count = get_fuzzy_count(regex, "rudy gobert", 5)
  # print "fuzzy_count for rudy gobert: %s || regex: %s" % (fuzzy_count, regex)
  # print "Name Confidence Level: %s" % lowest_fuzzy_count
  return lowest_fuzzy_player

def preprocess_predict(img, clf, pp, save = False, save_path = False):
  img = img.resize((30,30))
  img = img.filter(ImageFilter.SHARPEN)
  pic_data = np.array(img.convert('RGB'))
  pic_data = pic_data[:, :, ::-1].copy()
  save_data = pic_data
  pic_data = cv2.cvtColor(pic_data, cv2.COLOR_BGR2GRAY)
  pic_data = cv2.resize(pic_data, (30, 30), interpolation=cv2.INTER_AREA)
  pic_data = np.array(pic_data, 'int16')
  pic_hog_fd = hog(pic_data.reshape((30,30)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
  pic_hog_fd = pp.transform(np.array([pic_hog_fd], 'float64'))
  prediction = str(clf.predict(pic_hog_fd)[0])
  if save:
    cv2.imwrite(os.path.join(save_path, '%s_%s.png' % (prediction ,id_generator())), save_data)
  return prediction


ht_format_xy = {
  UNIVERSAL: {
    "HT_HEIGHT_RATIO": 30.0/960.0,
    "DIGIT_WIDTH_RATIO": 35.0/2.0/1707.0,
  },
  FORMAT1: {
    "HT_START_X_RATIO": 588.0/1707.0,
    "HT_START_Y_RATIO": 163.0/960.0,
  },
  FORMAT2: {
    "HT_START_X_RATIO": 588/1706.0,
    "HT_START_Y_RATIO": 198/960.0,
  }
}

def calc_height_rect(width, height, digit, card_format):
  start_x = ht_format_xy[card_format]["HT_START_X_RATIO"] * width
  start_y = ht_format_xy[card_format]["HT_START_Y_RATIO"] * height
  ht_height = ht_format_xy[UNIVERSAL]["HT_HEIGHT_RATIO"] * height
  digit_width = ht_format_xy[UNIVERSAL]["DIGIT_WIDTH_RATIO"] * width\
  # buffer width is to avoid the quotations in the height
  buffer_width = digit_width * (0.15) * (digit-1)
  x1 = start_x + digit_width * (digit-1) + buffer_width
  y1 = start_y
  x2 = x1 + digit_width * digit
  y2 = y1 + ht_height
  return (x1, y1, x2, y2)

def get_height(img, clf, pp, card_format, save=False, save_path=False):
  img = img.convert('L')
  img = change_contrast(img, 400)
  width, height = img.size
  # get feet of height
  img_ht = img.crop(calc_height_rect(width,height,1,card_format))
  prediction = preprocess_predict(img_ht, clf, pp, save, save_path)
  ht_guess = int(prediction) * 12
  # get inches of height
  img_ht = img.crop(calc_height_rect(width,height,2,card_format))
  prediction = preprocess_predict(img_ht, clf, pp, save, save_path)
  ht_guess += int(prediction)
  return ht_guess


def get_name(img, player_height, card_format, save=False):
  img = img.convert('L')
  img = change_contrast(img, 400)
  width, height = img.size
  good_guess_found = False
  for sharpen in range(0,2):
    if good_guess_found:
      break
    for y_adjust in range(0,16):
      img_player_name = img.crop(calc_name_rect(width, height, y_adjust, card_format))
      if sharpen > 0:
        img_player_name = img_player_name.filter(ImageFilter.SHARPEN)
      player_name_guess = pytesseract.image_to_string(img_player_name)
      # Remove all symbols and numbers
      player_name_guess = re.sub(r'[^a-zA-Z\s]', '', player_name_guess)
      # Remove all double spaces and new lines
      player_name_guess = re.sub(' +',' ', player_name_guess.replace('\n', ' ')).strip()
      split_player_name_guess = player_name_guess.split()
      if len(split_player_name_guess) >= 2 and len(split_player_name_guess) <= 3 and len(split_player_name_guess[0]) > 1:
        # img_player_name.show()
        good_guess_found = True
        break

  # Lower case
  player_name_guess = player_name_guess.lower()
  if save:
    img_player_name.save('Training/%s.png' % (player_name_guess))
  # load all players dict from json
  with open(os.path.join(BASE_DIR, 'players.json')) as fh:
    all_players_dict = json.load(fh)
  # Find closest match to names in json if player name guess is not an accurate player name
  original_player_name_guess = player_name_guess
  if len(player_name_guess) == 0:
    if save:
      img.save('Training/Bad_%s_%s.png' % (player_name_guess, id_generator()))
    return player_name_guess
  if player_name_guess not in all_players_dict:
    original_guess = player_name_guess
    player_name_guess = find_closest_nba_player_name(player_name_guess, player_height, all_players_dict)
  if save:
    img.save('Training/%s.png' % (player_name_guess))
  # Get original first and last name
  player_name = "%s %s" % (all_players_dict[player_name_guess]["firstName"], all_players_dict[player_name_guess]["lastName"])
  parse_logger.info("Original Name Guess: %s || Final Name %s" % (original_player_name_guess, player_name))
  return player_name

def get_stats(img, clf, pp, card_format, save=False, save_path=False):
  img = img.convert('L')
  img = change_contrast(img, 400)
  card_stats = {
    "(1,1)": {"name":"Speed", "value": 0},
    "(1,2)": {"name":"Agility", "value": 0},
    "(1,3)": {"name":"Mid-Range Shot", "value": 0},
    "(1,4)": {"name":"3 Point Shot", "value": 0},
    "(1,5)": {"name":"Inside Paint Shot", "value": 0},
    "(1,6)": {"name":"Post Shot", "value": 0},
    "(1,7)": {"name":"Dunking", "value": 0},
    "(1,8)": {"name":"Scoring With Contact", "value": 0},
    "(2,1)": {"name":"On Ball Defense", "value": 0},
    "(2,2)": {"name":"Block", "value": 0},
    "(2,3)": {"name":"Steal", "value": 0},
    "(2,4)": {"name":"Dribbling", "value": 0},
    "(2,5)": {"name":"Passing Accuracy", "value": 0},
    "(2,6)": {"name":"Box Out", "value": 0},
    "(2,7)": {"name":"Offensive Rebounding", "value": 0},
    "(2,8)": {"name":"Defensive Rebounding", "value": 0},
  }
  # Crop image to advanced stats portion
  for c in range(1,3):
    for r in range(1,9):
      digit = ""
      for d in range(1,3):
        width, height = img.size
        img_adv_stats = img.crop(calc_stat_rect(c,r,d,width,height, card_format))
        prediction = preprocess_predict(img_adv_stats, clf, pp, save, save_path)
        digit += prediction
      card_stats["(%s,%s)"%(c,r)]["value"] = int(digit) 
  return card_stats



ovr_format_xy = {
  UNIVERSAL: {
    "OVR_HEIGHT_RATIO": 60.0/960.0,
    "DIGIT_WIDTH_RATIO": 60.0/2.0/1707.0,
  },
  FORMAT1: {
    "OVR_START_X_RATIO": 458.0/1707.0,
    "OVR_START_Y_RATIO": 190.0/960.0,
  },
  FORMAT2: {
    "OVR_START_X_RATIO": 455.0/1706.0,
    "OVR_START_Y_RATIO": 225.0/960.0,
  }
}

def calc_ovr_rect(width, height, digit, card_format):
  start_x = ovr_format_xy[card_format]["OVR_START_X_RATIO"] * width
  start_y = ovr_format_xy[card_format]["OVR_START_Y_RATIO"] * height
  ovr_height = ovr_format_xy[UNIVERSAL]["OVR_HEIGHT_RATIO"] * height
  digit_width = ovr_format_xy[UNIVERSAL]["DIGIT_WIDTH_RATIO"] * width
  x1 = start_x + digit_width * 1.1 * (digit-1)
  y1 = start_y
  x2 = x1 + digit_width
  y2 = y1 + ovr_height
  return (x1, y1, x2, y2)

def get_ovr(img, clf, pp, card_format, save=False, save_path=False):
  img = img.convert('L')
  img = change_contrast(img, 400)
  width, height = img.size
  ovr_guess = ""
  for d in range(1,3):
    img_ovr = img.crop(calc_ovr_rect(width,height,d, card_format))
    prediction = preprocess_predict(img_ovr, clf, pp, save, save_path)
    ovr_guess += prediction
  if int(ovr_guess) < 50:
    return 0
  return int(ovr_guess)

card_xy_format = {
  UNIVERSAL: {
    "CARD_WIDTH_RATIO": 225.0/1706.0,
    "CARD_HEIGHT_RATIO": 335.0/960.0,
  },
  FORMAT1: {
    "CARD_START_X_RATIO": 310.0/1706.0,
    "CARD_START_Y_RATIO": 180.0/960.0,
  },
  FORMAT2: {
    "CARD_START_X_RATIO": 310.0/1706.0,
    "CARD_START_Y_RATIO": 215.0/960.0,
  }
}

def calc_card_rect(width, height, card_format):
  start_x = card_xy_format[card_format]["CARD_START_X_RATIO"] * width
  start_y = card_xy_format[card_format]["CARD_START_Y_RATIO"] * height
  card_width = card_xy_format[UNIVERSAL]["CARD_WIDTH_RATIO"] * width
  card_height = card_xy_format[UNIVERSAL]["CARD_HEIGHT_RATIO"] * height
  x1 = start_x
  y1 = start_y
  x2 = x1 + card_width
  y2 = y1 + card_height
  return (x1, y1, x2, y2)

def get_card_img(img, card_format, save=False, save_path=False):
  width, height = img.size
  img_card = img.crop(calc_card_rect(width, height, card_format))
  # img_card.show()
  output = StringIO()
  img_card.save(output, format='PNG')
  im_data = output.getvalue()
  # img_base64 = 'data:image/png;base64,' + base64.b64encode(im_data)
  img_base64 = base64.b64encode(im_data)
  return img_base64

pos_format_xy = {
  UNIVERSAL: {
    "POS_WIDTH_RATIO": 30.0/1706.0,
    "POS_HEIGHT_RATIO": 28.0/960.0,
  },
  FORMAT1: {
    "POS_START_X_RATIO": 473.0/1706.0,
    "POS_START_Y_RATIO": 268.0/960.0,
  },
  FORMAT2: {
    "POS_START_X_RATIO": 475.0/1706.0,
    "POS_START_Y_RATIO": 303.0/960.0,
  }
}

def calc_pos_rect(width, height, card_format):
  start_x = pos_format_xy[card_format]["POS_START_X_RATIO"] * width
  start_y = pos_format_xy[card_format]["POS_START_Y_RATIO"] * height
  pos_width = pos_format_xy[UNIVERSAL]["POS_WIDTH_RATIO"] * width
  pos_height = pos_format_xy[UNIVERSAL]["POS_HEIGHT_RATIO"] * height
  x1 = start_x
  y1 = start_y
  x2 = x1 + pos_width
  y2 = y1 + pos_height
  return (x1, y1, x2, y2)

def get_pos(img, clf, pp, card_format, save=False, save_path=False):
  img = img.convert('L')
  img = change_contrast(img, 400)
  width, height = img.size
  img_pos = img.crop(calc_pos_rect(width, height, card_format))
  prediction = preprocess_predict(img_pos, clf, pp, save, save_path)
  return prediction

type_format_xy = {
  UNIVERSAL: {
    "TYPE_WIDTH_RATIO": 65.0/1706.0,
    "TYPE_HEIGHT_RATIO": 65.0/960.0,
  },
  FORMAT1: {
    "TYPE_START_X_RATIO": 460.0/1706.0,
    "TYPE_START_Y_RATIO": 250.0/960.0,
  },
  FORMAT2: {
    "TYPE_START_X_RATIO": 360.0/1334.0,
    "TYPE_START_Y_RATIO": 225.0/750.0,
  }
}

def calc_type_rect(width, height, card_format):
  start_x = type_format_xy[card_format]["TYPE_START_X_RATIO"] * width
  start_y = type_format_xy[card_format]["TYPE_START_Y_RATIO"] * height
  type_width = type_format_xy[UNIVERSAL]["TYPE_WIDTH_RATIO"] * width
  type_height = type_format_xy[UNIVERSAL]["TYPE_HEIGHT_RATIO"] * height
  x1 = start_x
  y1 = start_y
  x2 = x1 + type_width
  y2 = y1 + type_height
  return (x1, y1, x2, y2)

def get_type(img, clf, pp, card_format, save=False, save_path=False):
  # img = img.convert('L')
  # img = change_contrast(img, 400)
  width, height = img.size
  pos_rect = calc_pos_rect(width, height, card_format)
  img.paste(0, [int(pos_rect[0]), int(pos_rect[1]), int(pos_rect[2]), int(pos_rect[3])])
  img_type = img.crop(calc_type_rect(width, height, card_format))
  prediction = preprocess_predict(img_type, clf, pp, save, save_path)
  return prediction


cancel_button_xy = {
  "X_RATIO": 800.0/1706.0,
  "Y_RATIO": 785.0/960.0,
  "WIDTH_RATIO": 120.0/1706.0,
  "HEIGHT_RATIO": 50.0/906.0,
}

def get_cancel_button_rect(width, height):
  start_x = cancel_button_xy["X_RATIO"] * width
  start_y = cancel_button_xy["Y_RATIO"] * height
  cancel_btn_width = cancel_button_xy["WIDTH_RATIO"] * width
  cancel_btn_height = cancel_button_xy["HEIGHT_RATIO"] * height
  x1 = start_x
  y1 = start_y
  x2 = x1 + cancel_btn_width
  y2 = y1 + cancel_btn_height
  return (x1, y1, x2, y2)

def get_format(img):
  """ Get format by checking if red cancel button exist"""
  img = img.convert('L')
  img = change_contrast(img, 400)
  width, height = img.size
  img_cancel_btn = img.crop(get_cancel_button_rect(width, height))
  # img_cancel_btn.show()
  cancel_btn_text = pytesseract.image_to_string(img_cancel_btn)
  if cancel_btn_text == "CANCEL":
    return FORMAT2
  return FORMAT1

def create_hash(pos, card_type, height, stats):
  card_dict_hash = {}
  card_dict_hash["pos"] = pos
  card_dict_hash["type"] = card_type
  card_dict_hash["height"] = height
  card_dict_hash["stats"] = stats
  return hashlib.sha1(json.dumps(card_dict_hash, sort_keys=True)).hexdigest()

def fix_ratio_if_needed(img):
  width, height = img.size
  ratio = round(width*1.0/height, 2)
  if ratio != 1.77 and ratio != 1.78:
    good_ratio = 1920*1.0/1080
    new_width = int(round(height * good_ratio, 0))
    img_new = Image.new('RGBA', (new_width, height), (255,255,255,255))
    img_new.paste(img, ((new_width-width)/2, 0))
    # img_new.show()
    return img_new
  return img

def parse_one(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp):
  card_dict = {
    "name": "",
    "ovr": 0,
    "type": "",
    "height": 0,
    "pos": "",
    "card_img": "",
    "stats": {},
    "hash": "",
  }

  img = fix_ratio_if_needed(img)
  card_format = get_format(img)

  card_dict["pos"] = get_pos(img, pos_clf, pos_pp, card_format)
  card_dict["card_img"] = get_card_img(img, card_format)
  card_dict["type"] = get_type(img, type_clf, type_pp, card_format)
  card_dict["height"] = get_height(img, height_clf, height_pp, card_format)
  card_dict["ovr"] = get_ovr(img, ovr_clf, ovr_pp, card_format)
  card_dict["name"] = get_name(img, card_dict["height"], card_format)
  card_dict["stats"] = get_stats(img, adv_stats_clf, adv_stats_pp, card_format) #, save=True, save_path="Training/adv_stats_digits")
  card_dict["hash"] = create_hash(card_dict["pos"], card_dict["type"], card_dict["height"], card_dict["stats"])
  return card_dict

def add_new_card_to_db(cards_db, card_dict):
  hashkey = card_dict["hash"]
  if not card_db.find_one({"hash": hashkey}):
    result = cards_db.insert_one(card_dict)
    return True
  return False

if __name__ == "__main__":
  # Load sk model
  adv_stats_clf, adv_stats_pp = joblib.load('Training/PKL/adv_stats_digits.pkl')
  height_clf, height_pp = joblib.load('Training/PKL/height_digits.pkl')
  ovr_clf, ovr_pp = joblib.load('Training/PKL/ovr_digits.pkl')
  pos_clf, pos_pp = joblib.load('Training/PKL/pos.pkl')
  type_clf, type_pp = joblib.load('Training/PKL/type.pkl')

  parser = OptionParser()
  parser.add_option("--img_path", dest="img_path", action="store")
  parser.add_option("--all", dest="all", action="store_true")
  (options, args) = parser.parse_args()

  parse_all = False
  img_path = "raw_cards\\271567.jpg"
  if options.img_path:
	  img_path = options.img_path
  if options.all:
	  parse_all = True

  if parse_all:
    all_players_dict = {}
    thread_count = 5
    tp = threadpool.ThreadPool(thread_count)
    threadpool_requests = []

    # import data into mongodb
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards

    def thread_func_wrapper(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp):
      card_dict = parse_one(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp)
      if card_dict["hash"] not in all_players_dict:
        all_players_dict["hash"] = card_dict
        add_new_card_to_db(cards_db, card_dict)
      return True

    for img_name in os.listdir("raw_cards"):
      img_path = os.path.join("raw_cards", img_name)
      img = Image.open(img_path)
      request = threadpool.WorkRequest(thread_func_wrapper,
                                      # underlying we use requests so (connect timeout, read timeout) both in seconds
                                      [img_path, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp],
                                      callback=threadpool.make_default_threadpool_callback(parse_logger),
                                      exc_callback=threadpool.make_default_threadpool_exception_callback(
                                          parse_logger),
                                      name='parse_card file_name=%s' % (img_name))
      threadpool_requests.append(request)

    for request in threadpool_requests:
      tp.putRequest(request)

    # to release some memory pressure
    del threadpool_requests[:]
    interrupted = False
    while True:
      try:
        tp.poll(block=True, timeout=20)
      except KeyboardInterrupt:
        interrupted = False
        break
        # we're done processing
      except threadpool.NoResultsPending:
        break
    tp.close()
  else:
    img = Image.open(img_path)
    card_dict = parse_one(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp)
	# Print out card's dict
    print "Name: %s" % card_dict["name"]
    print "OVR: %s" % card_dict["ovr"]
    print "Type: %s" % card_dict["type"]
    print "Height: %s" % card_dict["height"]
    print "Position: %s" % card_dict["pos"]
    print "Card Image: %s" % card_dict["card_img"][0:100]
    print "Hash: %s" % card_dict["hash"]
    for k, v in card_dict["stats"].iteritems():
      print "%s: %s" % (v["name"], v["value"])