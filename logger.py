'''
Default logging setup

Rev 20100702
This module encapsulates all the logging initialization and configuration options

@author: Sammy Yu
'''
import os
import sys
import logging
import logging.handlers
import multiprocessing
import multiprocessing.synchronize
import traceback
import threading


LOG_LEVEL_CRITICAL = logging.CRITICAL
LOG_LEVEL_ERROR = logging.ERROR
LOG_LEVEL_WARNING = logging.WARNING
LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_DEBUG = logging.DEBUG
LOG_LEVEL_NOTSET = logging.NOTSET

DEFAULT_LOG_LEVEL = int(os.getenv('LOGGING_LEVEL', LOG_LEVEL_INFO))


DEFAULT_LOG_FORMAT_STRING = "%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(message)s"

LOG_FORMAT_STRING_WITH_PID = "%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(process)s %(message)s"

LOG_FORMAT_STRING_WITH_THREAD_NAME = "%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(threadName)s %(message)s"

LOG_FORMAT_STRING_WITH_PID_AND_THREAD_NAME = "%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(process)s %(threadName)s %(message)s"

LOG_FORMAT_STRING_WITH_PID_COMPACT = "%(asctime)s (%(process)5d %(filename)s:%(lineno)4d) [%(levelname)3s] %(message)s"

# global dictionary  of logger, key = category, value=loggers
logger_dict = {}

class Singleton(type):

    def __init__(cls, name, bases, dict):
        #        print 'Calling Singleton.__init__'
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        #        print 'Calling Singleton.__call__'
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


class MultiProcessingLogFileHandler(logging.Handler):
    """multiprocessing log handler

    This handler makes it possible for several processes
    to log to the same file by using a queue.

    """

    def __init__(self, fname):
        logging.Handler.__init__(self)
        # just in case if we can't open file
        self._handler = None

        self._handler = logging.FileHandler(fname)
        self.queue = multiprocessing.Queue(-1)

        thrd = threading.Thread(target=self.receive)
        thrd.daemon = True
        thrd.start()

    def createLock(self):
        """
        Acquire a thread lock for serializing access to the underlying I/O.

        Override standard to use the multiprocessing.RLock
        to avoid deadlocks with standard RLock
        we typically would get this
#8 Frame 0x7fc0d8004a20, for file /usr/lib64/python2.7/logging/__init__.py, line 706, in acquire (self=<MultiProcessingLogFileHandler(_handler=<FileHandler(stream=<file at remote 0x7fc26b258b70>,
encoding=None, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc26a954cb0>, _RLock__count=0) at remote 0x7fc26a969590>, level=0, _name=None, delay=0, baseFilename='/var/log/brightedge/job_consumer.log', mode='a', filters=[], formatter=<Formatter(datefmt='%Y-%m-%d %H:%M:%S', _fmt='%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(message)s') at remote 0x7fc26a982410>) at remote 0x7fc26a969550>, level=20, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=140472956786432, _RLock__block=<thread.lock at remote 0x7fc26a954c90>, _RLock__count=1) at remote 0x7fc26a9694d0>, _name=None, queue=<Queue(_writer=<_multiprocessing.Connection at remote 0xf57070>, _recv=<built-in method recv of _multiprocessing.Connection object at remote 0xf56c40>, _thread=None, ...(truncated)
    self.lock.acquire()
#11 Frame 0x7fc1e4002cd0, for file /usr/lib64/python2.7/logging/__init__.py, line 755, in handle (self=<MultiProcessingLogFileHandler(_handler=<FileHandler(stream=
<file at remote 0x7fc26b258b70>, encoding=None, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc26a954cb0>, _RLock__count=0) at remote 0x7fc26a969590>, level=0, _name=None, delay=0, baseFilename='/var/log/brightedge/job_consumer.log', mode='a', filters=[], formatter=<Formatter(datefmt='%Y-%m-%d %H:%M:%S', _fmt='%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(message)s') at remote 0x7fc26a982410>) at remote 0x7fc26a969550>, level=20, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=140472956786432, _RLock__block=<thread.lock at remote 0x7fc26a954c90>, _RLock__count=1) at remote 0x7fc26a9694d0>, _name=None, queue=<Queue(_writer=<_multiprocessing.Connection at remote 0xf57070>, _recv=<built-in method recv of _multiprocessing.Connection object at remote 0xf56c40>, _thread=None, _...(truncated)
    self.acquire()
#14 Frame 0x7fc1940055e0, for file /usr/lib64/python2.7/logging/__init__.py, line 1334, in callHandlers (self=<Logger(name='crawl.db_url_list_parser', parent=<RootLogger(name='root', parent=None, handlers=[<StreamHandler(stream=<file at remote 0x7fc27adc11e0>, level=20, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc27ad72970>, _RLock__count=0) at remote 0x7fc26dfe9ad0>, _name=None, filters=[], formatter=<Formatter(datefmt=None, _fmt='%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(process)s %(threadName)s %(message)s') at remote 0x7fc26e69fd90>) at remote 0x7fc26dfe99d0>, <MultiProcessingLogFileHandler(_handler=<FileHandler(stream=<file at remote 0x7fc26b258b70>, encoding=None, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc26a954cb0>, _RLock__count=0) at remote 0x7fc26a969590>, level=0, _name=None, delay=0, baseFilename='/var/log/brightedge/job_consumer.log', mode='a', filters=[], fo...(truncated)
    hdlr.handle(record)
#17 Frame 0x7fc14c004a90, for file /usr/lib64/python2.7/logging/__init__.py, line 1294, in handle (self=<Logger(name='crawl.db_url_list_parser', parent=<RootLogger(name='root', parent=None, handlers=[<StreamHandler(stream=<file at remote 0x7fc27adc11e0>, level=20, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc27ad72970>, _RLock__count=0) at remote 0x7fc26dfe9ad0>, _name=None, filters=[], formatter=<Formatter(datefmt=None, _fmt='%(asctime)s (%(filename)s, %(funcName)s, %(lineno)d) [%(levelname)8s] %(process)s %(threadName)s %(message)s') at remote 0x7fc26e69fd90>) at remote 0x7fc26dfe99d0>, <MultiProcessingLogFileHandler(_handler=<FileHandler(stream=<file at remote 0x7fc26b258b70>, encoding=None, lock=<_RLock(_Verbose__verbose=False, _RLock__owner=None, _RLock__block=<thread.lock at remote 0x7fc26a954cb0>, _RLock__count=0) at remote 0x7fc26a969590>, level=0, _name=None, delay=0, baseFilename='/var/log/brightedge/job_consumer.log', mode='a', filters=[], formatte...(truncated)
        """
        self.lock = multiprocessing.RLock()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        if self._handler is not None:
            self._handler.close()
        logging.Handler.close(self)

class LoggerConfigurator(object):
    __metaclass__ = Singleton

    initialized = False
    disallow_skip_stderr_override = False
    skip_stderr = False

    def __init__(self, log_format=None, level=DEFAULT_LOG_LEVEL, base_handler_level=DEFAULT_LOG_LEVEL,
                 skip_stderr=None, multiprocessing=False):
        ''' Initial the the root logger format '''
        global DEFAULT_LOG_FORMAT_STRING

#        print 'SYU constructed loggerconfig with skip_stderr %s' % (skip_stderr)
#        import cStringIO
#        import inspect
#        stack_list = inspect.stack()
#        stack_trace = cStringIO.StringIO()
#        for stack in stack_list:
#            stack_trace.write('%s:%s %s\n' % (stack[1], stack[2], stack[3]))
#        print 'Calling LoggerConfigurator.__init__:%s with %s called from: %s' % (id(self), log_format, stack_trace.getvalue())
#        stack_trace.close()

        if LoggerConfigurator.initialized:
            #            print 'already initialized LoggerConfigurator.__init__:%s with %s' % (id(self), log_format)
            return

#        print 'Post check Calling LoggerConfigurator.__init__:%s with %s' % (id(self), log_format)

        # multiprocessing requires we change the lock
        self.multiprocessing = multiprocessing

        # set as None meaning leave as default, otherwise if it boolean we will set it
        if skip_stderr is not None:
            LoggerConfigurator.skip_stderr = skip_stderr

        if log_format is None:
            log_format = DEFAULT_LOG_FORMAT_STRING
        else:
            DEFAULT_LOG_FORMAT_STRING = log_format

        self.log_level = level
        logging.basicConfig(format=log_format, datefmt="%Y-%m-%d %H:%M:%S",
                            level=self.log_level)

        root_logger = self.get_top_level_logger()
        if len(root_logger.handlers) == 1:
            root_logger.handlers[0].setLevel(base_handler_level)

        if self.multiprocessing:
            self.reset_handlers_for_multiprocessing()

        LoggerConfigurator.initialized = True

    def reset_handlers_for_multiprocessing(self):
        root_logger = self.get_top_level_logger()
        for (index, handler) in enumerate(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) and \
                    not isinstance(handler, MultiProcessingLogFileHandler):
                if not isinstance(handler.lock, multiprocessing.synchronize.RLock):
                    handler.lock = multiprocessing.RLock()

    def override_log_format(self, log_format):
        '''
        Change log format
        '''
        global DEFAULT_LOG_FORMAT_STRING

        DEFAULT_LOG_FORMAT_STRING = log_format
        formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")

#        import cStringIO
#        import inspect
#        stack_list = inspect.stack()
#        stack_trace = cStringIO.StringIO()
#        for stack in stack_list:
#            stack_trace.write('%s:%s %s\n' % (stack[1], stack[2], stack[3]))
#        print 'Calling LoggerConfigurator.override_log_format:%s with %s called from: %s' % (id(self), log_format, stack_trace.getvalue())
#        stack_trace.close()

        root_logger = self.get_top_level_logger()
        for handler in root_logger.handlers:
            #            print 'Changing format for handler %s' % (handler)
            handler.setFormatter(formatter)

    def set_skip_stderr(self, skip_stderr):
        '''
        Avoid to skip writing to stderr
        '''
        # print 'SYU set skip stderr %s' % (skip_stderr)
        if LoggerConfigurator.disallow_skip_stderr_override:
            return
        LoggerConfigurator.skip_stderr = skip_stderr

    def set_disallow_skip_stderr_override(self, override):
        '''
        Don't allow for setting skip stderrr (needed for some command line script that import django)
        '''
        LoggerConfigurator.disallow_skip_stderr_override = override

    def set_base_handler_level(self, level):
        ''' This way so we don't spam the console for daemon processes. '''
        root_logger = self.get_top_level_logger()
        if len(root_logger.handlers) > 0:
            base_handler = root_logger.handlers[0]
            base_handler.setLevel(level)

    def get_logger(self, category):
        ''' Get the logger for the specified category '''
        global logger_dict

        logger = logging.getLogger(category)

        if category not in logger_dict:
            logger_dict[category] = logger
        return logger

    def set_boto_logger_level_to_warning(self):
        '''
        Set boto logger level to warning so we don't get too much spam
        '''
        botologger = self.get_logger('boto')
        botologger.setLevel(LOG_LEVEL_WARNING)

    def get_top_level_logger(self):
        return logging.getLogger()

    def add_file_handler(self, log_file_name, log_format=None,
                         auto_rotate=False, system_auto_rotate=False,
                         additional_loggers=None,
                         level=None, multiprocessing=False):
        '''
        @type  log_file_name       string
        @param log_file_name       the base of the log file name
        @type  log_format          string
        @param log_format          the format of each log line
        @type  auto_rotate         boolean
        @param auto_rotate         whether or not to automatically have python logging facilities
                                   to rotate the logs.  If the log file is not rotated here
                                   we should use the system log rotater.
        @type  system_auto_rotate  boolean
        @param system_auto_rotate  whether or not to be aware of the fact that the system is
                                   rotating the logs (only applicable to mediators that is a
                                   daemon that is always running).
        @type  multiprocessing     boolean
        @param multiprocessing     logger is safe is for multiprocessing
        @type additional_loggers   list
        @type level                int
        '''
        # system auto rotate and auto rate are mututally exclusive
        if system_auto_rotate and os.name != 'nt':
            fh = logging.handlers.WatchedFileHandler(log_file_name)
        elif auto_rotate:
            # todo haven't implemented rotate version of multiprocessing safe log file handler so use the non-rotating version
            if multiprocessing:
                fh = MultiProcessingLogFileHandler(log_file_name)
            else:
                fh = logging.handlers.TimedRotatingFileHandler(log_file_name, 'midnight', 1, backupCount=9)
        else:
            if multiprocessing:
                fh = MultiProcessingLogFileHandler(log_file_name)
            else:
                fh = logging.FileHandler(log_file_name)

        self.add_log_handler(fh, log_format, additional_loggers, level)

    def add_log_handler(self, log_handler, log_format=None, additional_loggers=None, level=None):
        '''
        @type  log_handler       log handler object
        @param log_handler       log handler
        @type  log_format        string
        @param log_format        the format of each log line
        @type additional_loggers list
        @type  level              it
        '''
        global DEFAULT_LOG_FORMAT
        global logger_dict

        root_logger = logging.getLogger()
        if log_format is None:
            log_format = DEFAULT_LOG_FORMAT_STRING

        formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
        if level is None:
            level = DEFAULT_LOG_LEVEL
        log_handler.setLevel(level)
        log_handler.setFormatter(formatter)

        # now putting in root logger
        root_logger.addHandler(log_handler)

        # set from construction
        # @see http://stackoverflow.com/questions/2266646/how-to-i-disable-and-re-enable-console-logging-in-python
#        print 'SYU self.skip_stderr %s' % (self.skip_stderr)
        if self.skip_stderr:
            handlers_to_remove = []
            for (index, handler) in enumerate(root_logger.handlers):
                #                root_logger.fatal('ZZZ found handler #%s:%s' % (index, fh))
                # remove only if it is sys.<....> streamhandler
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) and \
                        not isinstance(handler, MultiProcessingLogFileHandler):
                    # root_logger.fatal('ZZZ found handle to remove #%s:%s class %s stream %s'
                    # % (index, handler, handler.__class__, handler.stream))
                    handlers_to_remove.append(handler)
            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)

        # as well as additional loggers
        if additional_loggers is not None:
            for logger_name in additional_loggers:
                logger_obj = logging.getLogger(logger_name)
                logger_obj.addHandler(fh)
#        for logger in logger_dict.values():
#            logger.addHandler(fh)
