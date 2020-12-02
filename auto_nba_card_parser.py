from nba_card_parser import parse_one, parse_one_main_stats, add_new_card_to_db, check_if_card_exist_in_db
from pymongo import MongoClient
from sklearn.externals import joblib
from optparse import OptionParser
import PIL
from PIL import Image, ImageFilter
from itertools import izip

import glob
import os
import json
import threadpool
import traceback
import time

import logger
logconfig = logger.LoggerConfigurator()
auto_parse_logger = logconfig.get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)


def check_card_errors(card_main_data, card_dict):
    for index in card_dict["stats"]:
        stat = card_dict["stats"][index]
        if stat["value"] < 15:
            return True
    if card_dict["ovr"] < 70 or card_dict["ovr"] > 99:
        return True
    if card_dict["height"] < 66:
        return True
    for value in ["ovr", "name", "type", "pos"]:
        if card_main_data[value] != card_dict[value]:
            return True
    return False


img_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Screenshots\\"
processing_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Processing Screenshots\\"
completed_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Completed Screenshots\\"
duplicate_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Duplicate Screenshots\\"
error_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Error Screenshots\\"
exception_path = "C:\\Users\\hkuang\\Pictures\\MEmu Photo\\Exception Screenshots\\"

client = MongoClient("mongodb://henleyk:test1234@nbalivemobilecards-shard-00-00.bp4sq.mongodb.net:27017,nbalivemobilecards-shard-00-01.bp4sq.mongodb.net:27017,nbalivemobilecards-shard-00-02.bp4sq.mongodb.net:27017/nbalivemobilecards?ssl=true&replicaSet=atlas-zdv1z0-shard-0&authSource=admin&retryWrites=true&w=majority")
db = client.nbalivemobilecards
cards_db = db.cards2


def thread_func_wrapper(files, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp, dry):
    # get files
    ss_1 = os.path.basename(files[0])
    ss_2 = os.path.basename(files[1])
    auto_parse_logger.info("Processing: %s || %s" % (ss_1, ss_2))
    # move files
    processing_path_1 = os.path.join(processing_path, ss_1)
    os.rename(files[0], processing_path_1)
    processing_path_2 = os.path.join(processing_path, ss_2)
    os.rename(files[1], processing_path_2)
    rename_file = None
    try:
        # parse adv stats img
        img = Image.open(processing_path_2)
        card_dict = parse_one(img, adv_stats_clf, adv_stats_pp, height_clf,
                              height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp, dry)
        # parse main stats img
        img = Image.open(processing_path_1)
        card_main_data = parse_one_main_stats(
            img, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp, card_dict["height"])
        # rename to <name>__<ovr>_<type>_<hash>
        rename_file = "%s_%s_%s_%s" % (card_dict["name"].replace(
            " ", "_"), card_dict["ovr"], card_dict["type"], card_dict["hash"])
        if check_card_errors(card_main_data, card_dict):
            error_path_1 = "%s%s_1.png" % (error_path, rename_file)
            if not os.path.exists(error_path_1):
                os.rename(processing_path_1, error_path_1)
            else:
                os.remove(processing_path_1)
            error_path_2 = "%s%s_2.png" % (error_path, rename_file)
            if not os.path.exists(error_path_2):
                os.rename(processing_path_2, error_path_2)
            else:
                os.remove(processing_path_2)
            auto_parse_logger.info("Error: %s || %s" %
                                   (error_path_1, error_path_2))
        elif options.dry:
            pass
        elif add_new_card_to_db(cards_db, card_dict):
            completed_path_1 = "%s%s_1.png" % (completed_path, rename_file)
            os.rename(processing_path_1, completed_path_1)
            completed_path_2 = "%s%s_2.png" % (completed_path, rename_file)
            os.rename(processing_path_2, completed_path_2)
            auto_parse_logger.info("Completed: %s || %s" %
                                   (completed_path_1, completed_path_2))
        else:
            auto_parse_logger.info("Duplicate Found.")
            duplicate_path_1 = "%s%s_1.png" % (duplicate_path, rename_file)
            if not os.path.exists(duplicate_path_1):
                os.rename(processing_path_1, duplicate_path_1)
            else:
                os.remove(processing_path_1)
            duplicate_path_2 = "%s%s_2.png" % (duplicate_path, rename_file)
            if not os.path.exists(duplicate_path_2):
                os.rename(processing_path_2, duplicate_path_2)
            else:
                os.remove(processing_path_2)
    except Exception:
        auto_parse_logger.info("Exception: %s" % traceback.format_exc())
        if not rename_file:
            exception_path_1 = "%s%s_1.png" % (exception_path, ss_1)
            exception_path_2 = "%s%s_1.png" % (exception_path, ss_2)
        else:
            exception_path_1 = "%s%s_1.png" % (exception_path, rename_file)
            exception_path_2 = "%s%s_2.png" % (exception_path, rename_file)
        os.rename(processing_path_1, exception_path_1)
        os.rename(processing_path_2, exception_path_2)
    return True


if __name__ == "__main__":
    # Load sk model
    adv_stats_clf, adv_stats_pp = joblib.load(
        os.path.join(BASE_DIR, 'Training/PKL/adv_stats_digits.pkl'))
    height_clf, height_pp = joblib.load(os.path.join(
        BASE_DIR, 'Training/PKL/height_digits.pkl'))
    ovr_clf, ovr_pp = joblib.load(os.path.join(
        BASE_DIR, 'Training/PKL/ovr_digits.pkl'))
    pos_clf, pos_pp = joblib.load(
        os.path.join(BASE_DIR, 'Training/PKL/pos.pkl'))
    type_clf, type_pp = joblib.load(
        os.path.join(BASE_DIR, 'Training/PKL/type.pkl'))

    parser = OptionParser()
    parser.add_option("--dry", dest="dry", action="store_true")
    (options, args) = parser.parse_args()

    thread_count = 5
    tp = threadpool.ThreadPool(thread_count)
    threadpool_requests = []

    files = glob.glob("%s*.png" % img_path)
    sorted(files)
    file_count = len(files)
    if file_count % 2 != 0:
        auto_parse_logger.info("There is an odd number of files, exiting")
        sys.exit()
    for file1, file2 in pairwise(files):
        both_files = [file1, file2]
        request = threadpool.WorkRequest(thread_func_wrapper,
                                         # underlying we use requests so (connect timeout, read timeout) both in seconds
                                         [both_files, adv_stats_clf, adv_stats_pp, height_clf, height_pp,
                                             ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp, options.dry],
                                         callback=threadpool.make_default_threadpool_callback(
                                             auto_parse_logger),
                                         exc_callback=threadpool.make_default_threadpool_exception_callback(
                                             auto_parse_logger),
                                         name='parse_card thread %s' % file1)
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
