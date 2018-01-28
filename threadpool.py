# -*- coding: UTF-8 -*-
"""Easy to use object-oriented thread pool framework.

A thread pool is an object that maintains a pool of worker threads to perform
time consuming operations in parallel. It assigns jobs to the threads
by putting them in a work request queue, where they are picked up by the
next available thread. This then performs the requested operation in the
background and puts the results in another queue.

The thread pool object can then collect the results from all threads from
this queue as soon as they become available or after all threads have
finished their work. It's also possible, to define callbacks to handle
each result as it comes in.

The basic concept and some code was taken from the book "Python in a Nutshell,
2nd edition" by Alex Martelli, O'Reilly 2006, ISBN 0-596-10046-9, from section
14.5 "Threaded Program Architecture". I wrapped the main program logic in the
ThreadPool class, added the WorkRequest class and the callback system and
tweaked the code here and there. Kudos also to Florent Aide for the exception
handling mechanism.

Basic usage::

    >>> pool = ThreadPool(poolsize)
    >>> requests = makeRequests(some_callable, list_of_args, callback)
    >>> [pool.putRequest(req) for req in requests]
    >>> pool.wait()

See the end of the module code for a brief, annotated usage example.

Website : http://chrisarndt.de/projects/threadpool/

"""
__docformat__ = "restructuredtext en"

__all__ = [
    'makeRequests',
    'NoResultsPending',
    'NoWorkersAvailable',
    'ThreadPool',
    'WorkRequest',
    'WorkerThread'
]

__author__ = "Christopher Arndt"
__version__ = '1.2.7'
__revision__ = "$Revision: 416 $"
__date__ = "$Date: 2009-10-07 05:41:27 +0200 (Wed, 07 Oct 2009) $"
__license__ = "MIT license"


# standard library modules
import sys
import threading
import Queue
import traceback
import multiprocessing

# the work the threads will have to do (rather trivial in our example)
def do_something(data):
    import time
    import random
    time.sleep(random.randint(1, 5))
    result = round(random.random() * data, 5)
    # just to show off, we throw an exception once in a while
    if result > 5:
        raise RuntimeError("Something extraordinary happened!")
    return result


THREADPOOL_THREAD_MODE = 0
THREADPOOL_FORK_MODE = 1
THREADPOOL_MAX_SUBMIT_RETRY_ATTEMPTS = 5
# exceptions
class NoResultsPending(Exception):
    """All work requests have been processed."""
    pass

class NoWorkersAvailable(Exception):
    """No worker threads available to process remaining requests."""
    pass


# internal module helper functions
def _handle_thread_exception(threadpool, request, exc_info):
    """Default exception handler callback function.

    This just prints the exception info via ``traceback.print_exception``.

    """
    traceback.print_exception(*exc_info)


# utility functions
def makeRequests(callable_, args_list, callback=None,
                 exc_callback=_handle_thread_exception):
    """Create several work requests for same callable with different arguments.

    Convenience function for creating several work requests for the same
    callable where each invocation of the callable receives different values
    for its arguments.

    ``args_list`` contains the parameters for each invocation of callable.
    Each item in ``args_list`` should be either a 2-item tuple of the list of
    positional arguments and a dictionary of keyword arguments or a single,
    non-tuple argument.

    See docstring for ``WorkRequest`` for info on ``callback`` and
    ``exc_callback``.

    """
    requests = []
    for item in args_list:
        if isinstance(item, tuple):
            requests.append(
                WorkRequest(callable_, item[0], item[1], callback=callback,
                            exc_callback=exc_callback)
            )
        else:
            requests.append(
                WorkRequest(callable_, [item], None, callback=callback,
                            exc_callback=exc_callback)
            )
    return requests


def multiprocessing_target_wrapper(callable, queue, args, kwargs):
    '''
    This is a callable target that is used by multiprocess to wrap around the real callable
    It is necessary in order to capture the results in the queue
    '''
    result = None
    try:
        result = callable(*args, **kwargs)
    except:
        pass
    finally:
        queue.put(result)
    sys.exit(0)

# classes
class WorkerThread(threading.Thread):
    """Background thread connected to the requests/results queues.

    A worker thread sits in the background and picks up work requests from
    one queue and puts the results in another until it is dismissed.

    """

    def __init__(self, requests_queue, results_queue, poll_timeout=5,
                 mode=THREADPOOL_THREAD_MODE, **kwds):
        """Set up thread in daemonic mode and start it immediatedly.

        ``requests_queue`` and ``results_queue`` are instances of
        ``Queue.Queue`` passed by the ``ThreadPool`` class when it creates a new
        worker thread.

        """
        threading.Thread.__init__(self, **kwds)
        self.setDaemon(1)
        self._requests_queue = requests_queue
        self._results_queue = results_queue
        self._poll_timeout = poll_timeout
        self._dismissed = threading.Event()
        self.mode = mode
        self.start()

    def run(self):
        """Repeatedly process the job queue until told to exit."""
        while True:
            if self._dismissed.isSet():
                # we are dismissed, break out of loop
                break
            # get next work request. If we don't get a new request from the
            # queue after self._poll_timout seconds, we jump to the start of
            # the while loop again, to give the thread a chance to exit.
            try:
                request = self._requests_queue.get(True, self._poll_timeout)
                threading.currentThread().setName(request.name)
            except Queue.Empty:
                continue
            else:
                if self._dismissed.isSet():
                    # we are dismissed, put back request in queue and exit loop
                    self._requests_queue.put(request)
                    break
                try:
                    if self.mode == THREADPOOL_FORK_MODE:
                        q = multiprocessing.Queue()
                        p = multiprocessing.Process(name=request.name,
                                                    target=multiprocessing_target_wrapper, args=[request.callable, q, request.args, request.kwds])
                        p.start()
                        p.join()
                        result = q.get()
                    else:
                        result = request.callable(*request.args, **request.kwds)
                    self._results_queue.put((request, result))
                    # SYU clear out call-able to conserve memory
                    request.callable = None
                except:
                    request.exception = True
                    self._results_queue.put((request, sys.exc_info()))

    def dismiss(self):
        """Sets a flag to tell the thread to exit when done with current job."""
        self._dismissed.set()


class WorkRequest(object):
    """A request to execute a callable for putting in the request queue later.

    See the module function ``makeRequests`` for the common case
    where you want to build several ``WorkRequest`` objects for the same
    callable but with different arguments for each call.

    """

    def __init__(self, callable_, args=None, kwds=None, name=None, requestID=None,
                 callback=None, exc_callback=_handle_thread_exception):
        """Create a work request for a callable and attach callbacks.

        A work request consists of the a callable to be executed by a
        worker thread, a list of positional arguments, a dictionary
        of keyword arguments.

        A ``callback`` function can be specified, that is called when the
        results of the request are picked up from the result queue. It must
        accept two anonymous arguments, the ``WorkRequest`` object and the
        results of the callable, in that order. If you want to pass additional
        information to the callback, just stick it on the request object.

        You can also give custom callback for when an exception occurs with
        the ``exc_callback`` keyword parameter. It should also accept two
        anonymous arguments, the ``WorkRequest`` and a tuple with the exception
        details as returned by ``sys.exc_info()``. The default implementation
        of this callback just prints the exception info via
        ``traceback.print_exception``. If you want no exception handler
        callback, just pass in ``None``.

        ``requestID``, if given, must be hashable since it is used by
        ``ThreadPool`` object to store the results of that work request in a
        dictionary. It defaults to the return value of ``id(self)``.

        """
        self.name = name
        if requestID is None:
            self.requestID = id(self)
        else:
            try:
                self.requestID = hash(requestID)
            except TypeError:
                raise TypeError("requestID must be hashable.")
        self.exception = False
        self.callback = callback
        self.exc_callback = exc_callback
        self.callable = callable_
        self.args = args or []
        self.kwds = kwds or {}

    def __str__(self):
        return "<WorkRequest id=%s args=%r kwargs=%r exception=%s>" % \
            (self.requestID, self.args, self.kwds, self.exception)

def make_default_threadpool_callback(logger):
    '''
    A default implementation of a callback for threadpool requests
    This callback is executed when a request has completed
    '''
    def threadpool_callback(threadpool, request, result):
        if result is False:
            logger.error('Thread request %s failed, aborting thread pool' % (request.name))
            threadpool.return_code = False
            # dismiss work is only available on non futures threadpool
            if not isinstance(threadpool, FuturesThreadPool):
                threadpool.dismissWorkers(0)
        else:
            logger.info('Thread request %s completed successfully' % (request.name))
        return
    return threadpool_callback

def make_default_threadpool_exception_callback(logger):
    '''
    A default implementation of a exception callback for threadpool requests
    This callback is executed when a request in thread pool has thrown exception
    '''
    def threadpool_exception_callback(threadpool, request, exception):
        '''
        @param threadpool threadpool of the worker
        @param request    Worker Request
        @param exception  is a tuple of 3 (type, value, traceback)
        '''
        exception_string_list = traceback.format_exception(*exception)
        logger.error('Thread request %s failed got an exception: [%s]' % (request.name, ''.join(exception_string_list)))
        threadpool.return_code = False
        # dismiss work is only available on non futures threadpool
        if not isinstance(threadpool, FuturesThreadPool):
            threadpool.dismissWorkers(0)
    return threadpool_exception_callback

def collect_failure_from_return_value_threadpool_callback(cacher_logger, failed_list, failure_unit):
    '''
    A default implementation of a callback for threadpool requests
    This callback is executed when a request has completed
    '''
    def threadpool_callback(threadpool, request, result):
        if result is False:
            cacher_logger.error('Thread request %s failed, aborting thread pool' % (request.name))
            failed_list.append(failure_unit)
            threadpool.return_code = False
            if not isinstance(threadpool, FuturesThreadPool):
                threadpool.dismissWorkers(0)
        else:
            threadpool.return_code = True
            cacher_logger.info('Thread request %s completed successfully' % (request.name))
        return result
    return threadpool_callback

def collect_failure_from_exception_callback(logger, failed_list, failure_unit):
    '''
    A default implementation of a exception callback for threadpool requests
    This callback is executed when a request in thread pool has thrown exception
    '''
    def threadpool_exception_callback(threadpool, request, exception):
        '''
        @param threadpool threadpool of the worker
        @param request    Worker Request
        @param exception  is a tuple of 3 (type, value, traceback)
        '''
        exception_string_list = traceback.format_exception(*exception)
        logger.error('Thread request %s failed got an exception: [%s]' % (request.name, ''.join(exception_string_list)))
        failed_list.append(failure_unit)
        threadpool.return_code = False

        if not isinstance(threadpool, FuturesThreadPool):
            threadpool.dismissWorkers(0)

    return threadpool_exception_callback

class ThreadPool(object):
    """
    A thread pool, distributing work requests and collecting results.

    See the module docstring for more information.
    """

    def __init__(self, num_workers, q_size=0, resq_size=0, poll_timeout=5, mode=THREADPOOL_THREAD_MODE):
        """Set up the thread pool and start num_workers worker threads.

        ``num_workers`` is the number of worker threads to start initially.

        If ``q_size > 0`` the size of the work *request queue* is limited and
        the thread pool blocks when the queue is full and it tries to put
        more work requests in it (see ``putRequest`` method), unless you also
        use a positive ``timeout`` value for ``putRequest``.

        If ``resq_size > 0`` the size of the *results queue* is limited and the
        worker threads will block when the queue is full and they try to put
        new results in it.

        .. warning:
            If you set both ``q_size`` and ``resq_size`` to ``!= 0`` there is
            the possibility of a deadlock, when the results queue is not pulled
            regularly and too many jobs are put in the work requests queue.
            To prevent this, always set ``timeout > 0`` when calling
            ``ThreadPool.putRequest()`` and catch ``Queue.Full`` exceptions.

        """
        # a helper attribute to help determine the final status of the thread pool
        # the call back functions can manipulate this
        self.return_code = True
        self.total_requests = None
        self.requests_completed = 0

        self._requests_queue = Queue.Queue(q_size)
        self._results_queue = Queue.Queue(resq_size)
        self.workers = []
        self.dismissedWorkers = []
        self.workRequests = {}
        self.mode = mode
        self.createWorkers(num_workers, poll_timeout)

    def createWorkers(self, num_workers, poll_timeout=5):
        """Add num_workers worker threads to the pool.

        ``poll_timout`` sets the interval in seconds (int or float) for how
        ofte threads should check whether they are dismissed, while waiting for
        requests.

        """
        for _ in range(num_workers):
            worker = None
            worker = WorkerThread(self._requests_queue, self._results_queue, poll_timeout=poll_timeout,
                                  mode=self.mode)
            self.workers.append(worker)

    def dismissWorkers(self, num_workers, do_join=False):
        """Tell num_workers worker threads to quit after their current task."""
        dismiss_list = []
        for _ in range(min(num_workers, len(self.workers))):
            worker = self.workers.pop()
            worker.dismiss()
            dismiss_list.append(worker)

        if do_join:
            for worker in dismiss_list:
                worker.join()
        else:
            self.dismissedWorkers.extend(dismiss_list)

    def joinAllDismissedWorkers(self):
        """Perform Thread.join() on all worker threads that have been dismissed.
        """
        for worker in self.dismissedWorkers:
            worker.join()
        self.dismissedWorkers = []

    def putRequest(self, request, block=True, timeout=None):
        """Put work request into work queue and save its id for later."""
        assert isinstance(request, WorkRequest)
        # don't reuse old work requests
        assert not getattr(request, 'exception', None)
        # Add retry logic here to handle the Exception which happens when creating lots of child processes in a short time
        attempt_count = 0
        while attempt_count < THREADPOOL_MAX_SUBMIT_RETRY_ATTEMPTS:
            try:
                self._requests_queue.put(request, block, timeout)
                break
            except Exception as e:
                print("Failed to add request %s to threadpool #%s on attempt : %s" % (str(request), str(attempt_count), str(e)))
                time.sleep(1)
                attempt_count = attempt_count + 1
                if attempt_count == THREADPOOL_MAX_SUBMIT_RETRY_ATTEMPTS:
                    print("Could not add request %s to threadpool: %s" % (str(request), str(e)))
                    raise e
        self.workRequests[request.requestID] = request

    def getWorkRequests(self):
        '''Lists out the work requests in queue'''
        return self.workRequests

    def calculateTotalRequests(self):
        '''
        Get and set the total number of requests
        '''
        self.total_requests = len(self.workRequests)
        return self.total_requests

    def getRemainingRequests(self):
        '''
        Get and number of jobs currently running and queued (not completed)
        '''
        return len(self.workRequests)

    def poll(self, block=False, timeout=None,
             logger=None, total_requests=None, print_every_requests=100):
        """
        Process any new results in the queue.
        """

        if total_requests is not None:
            self.total_requests = total_requests

        while True:
            # still results pending?
            if not self.workRequests:
                raise NoResultsPending
            # are there still workers to process remaining requests?
            elif block and not self.workers:
                raise NoWorkersAvailable
            try:
                # get back next results
                request, result = self._results_queue.get(block=block,
                                                          timeout=timeout)
                self.requests_completed = self.requests_completed + 1
                if logger is not None:
                    if self.requests_completed % print_every_requests == 0:
                        log_msg = '[Threadpool completed: %s]' % self.requests_completed
                        if self.total_requests is not None:
                            perc = float(self.requests_completed) / self.total_requests * 100
                            log_msg = '[Threadpool completed: %s/%s: %0.2f%%]' % (self.requests_completed, self.total_requests, perc)
                        logger.info(log_msg)
                # has an exception occurred?
                if request.exception and request.exc_callback:
                    request.exc_callback(self, request, result)
                # hand results to callback, if any
                if request.callback and not \
                        (request.exception and request.exc_callback):
                    request.callback(self, request, result)
                del self.workRequests[request.requestID]
            except Queue.Empty:
                break

    def wait(self):
        """Wait for results, blocking until all have arrived."""
        while True:
            try:
                self.poll(True)
            except NoResultsPending:
                break

    def close(self):
        """
        Dismiss all the remaining workers
        """
        self.dismissWorkers(len(self.workers), True)

    @staticmethod
    def set_thread_stack_size(bytes):
        """
        Change the thread stack size
        Defaults to 10mb on linux which may be a bit too much in some cases
        """
        threading.stack_size(bytes)

# needed for now since we dont have python-futures on CentOS 5
# used by cla
try:
    import concurrent.futures

    class FuturesThreadPool(ThreadPool):
        '''
        ThreadPool layer on top of pythonfutures (https://code.google.com/p/pythonfutures/), default threadpool library in Python 3.2
        '''

        def __init__(self, num_workers, q_size=0, resq_size=0, poll_timeout=5, mode=THREADPOOL_THREAD_MODE):
            # a helper attribute to help determine the final status of the thread pool
            # the call back functions can manipulate this
            self.return_code = True
            self.future_dict = {}
            self.requests_completed = 0
            self.total_requests = None

            if mode == THREADPOOL_THREAD_MODE:
                self.futures_threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
            else:
                self.futures_threadpool = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)

        def getWorkRequests(self):
            '''Lists out the work requests in queue'''
            return self.future_dict

        def putRequest(self, request, block=True, timeout=None):
            """
            Put work request into work queue and save its id for later.
            """
            assert isinstance(request, WorkRequest)

            # Add retry logic here to handle the Exception which happens when creating lots of child processes in a short time
            attempt_count = 0
            while attempt_count < THREADPOOL_MAX_SUBMIT_RETRY_ATTEMPTS:
                try:
                    request_future = self.futures_threadpool.submit(request.callable, *request.args, **request.kwds)
                    break
                except Exception as e:
                    print("Failed to add request %s to threadpool #%s on attempt : %s" % (str(request), str(attempt_count), str(e)))
                    time.sleep(1)
                    attempt_count = attempt_count + 1
                    if attempt_count == THREADPOOL_MAX_SUBMIT_RETRY_ATTEMPTS:
                        print("Could not add request %s to threadpool: %s" % (str(request), str(e)))
                        raise e
            self.future_dict[request_future] = request

        def getWorkRequests(self):
            '''Lists out the work requests in queue'''
            return self.future_dict

        def getRemainingRequests(self):
            '''
            Get and number of jobs currently running and queued (not completed)
            '''
            return len(self.future_dict)

        def calculateTotalRequests(self):
            '''
            Get and set the total number of requests
            '''
            self.total_requests = len(self.future_dict)
            return self.total_requests

        def minipoll(self, logger):
            as_completed_timeout = 5
            timedout_count = 0
#            future_generator = concurrent.futures.as_completed(self.future_dict, timeout=as_completed_timeout)
#            try:
#                for (request_index, request_future) in enumerate(future_generator):
#                    logger.error('[SYU] Polling for request future #%s:%s ITERATE' % (request_index, str(request_future)))
#            except concurrent.futures.TimeoutError:
#                # timeout specified for poll, the futures object is still processing past timeout
#                logger.error('[SYU] timed out getting next')
#                return
            for (future, req) in self.getWorkRequests().items():
                done = future.done()
                running = future.running()
                cancelled = future.cancelled()
                logger.info('Request remaining in queue checking return for: %s, status=%s/%s/%s' % (req.name,
                                                                                                     done, running, cancelled))
                try:
                    data = future.result(timeout=as_completed_timeout)
                except concurrent.futures.TimeoutError:
                    logger.info('Request timed-out remaining in queue checking return for: %s' % (req.name))
                    timedout_count = timedout_count + 1
                    if timedout_count >= 5:
                        break

        def poll(self, block=False, timeout=None,
                 logger=None, total_requests=None, print_every_requests=100):
            """
            Process any new results in the queue.
            if block is True and timeout is None, poll will block indefinitely until there is a result
              if timeout is a positive number, it blocks at most timeout seconds
            if block is False, process an item immediately if one is avaialble otherwise return, timeout is ignored


            @param block      whether or not to block until we get a result
            @param timeout    the maximum number of seconds we will wait for a result
            """

            if total_requests is not None:
                self.total_requests = total_requests

            as_completed_timeout = None
            if block and timeout is not None and timeout > 0:
                as_completed_timeout = timeout
            elif block == False:
                as_completed_timeout = 0.1

            future_generator = concurrent.futures.as_completed(self.future_dict, timeout=as_completed_timeout)
            try:
                for (request_index, request_future) in enumerate(future_generator):
                    if logger is not None:
                        logger.debug('[SYU] Polling for request future #%s:%s ITERATE' % (request_index, str(request_future)))
                    try:
                        request = self.future_dict[request_future]
                    except KeyError:
                        # likely killed off with close
                        # KeyError: <Future at 0x4ac7e50 state=finished returned bool>
                        # if self.future_dict.has_key(request_future):
                        #    del self.future_dict[request_future]
                        # self.requests_completed = self.requests_completed + 1
                        continue
                    # get back next results
                    try:
                        if logger is not None:
                            logger.debug('[SYU] Polling for request future #%s:%s START' % (request_index, str(request_future)))
                        if block:
                            if timeout is not None and timeout > 0:
                                data = request_future.result(timeout=timeout)
                            else:
                                data = request_future.result()
                        else:
                            data = request_future.result(timeout=0.1)
                    except concurrent.futures.TimeoutError:
                        break
                    except Exception:
                        if request.exc_callback:
                            # Hard for concurrent futures to keep track of original stack trace:
                            # http://stackoverflow.com/questions/19309514/how-to-get-correct-line-number-where-exception-was-thrown-using-concurrent-futur
                            # fixed now in futures 2.2.0+ https://github.com/danielj7/pythonfutures/commit/0f7e6dd16d3cd0307b90bdfc7dde7ececeaef857
                            exception_tuple = sys.exc_info()
                            if hasattr(request, 'exception_info'):
                                exception_info = request.exception_info()
                                exception = exception_info[0]
                                exception_traceback = exception_info[1]
                                exception_tuple = (exception.__class__, exception, exception_traceback)
                            request.exc_callback(self, request, exception_tuple)
                        request.exception = True
                    finally:
                        if logger is not None:
                            logger.debug('[SYU] Polling for request future #%s:%s DONE' % (request_index, str(request_future)))

                    # hand results to callback, if any
                    if request.callback and not \
                            (request.exception and request.exc_callback):
                        request.callback(self, request, data)
                    del self.future_dict[request_future]
                    self.requests_completed = self.requests_completed + 1
                    if logger is not None:
                        if self.requests_completed % print_every_requests == 0:
                            log_msg = '[Threadpool completed: %s]' % self.requests_completed
                            if self.total_requests is not None:
                                perc = float(self.requests_completed) / self.total_requests * 100
                                log_msg = '[Threadpool completed: %s/%s: %0.2f%%]' % (self.requests_completed, self.total_requests, perc)
                            logger.info(log_msg)
                raise NoResultsPending
            except concurrent.futures.TimeoutError:
                # timeout specified for poll, the futures object is still processing past timeout
                return

        def wait(self):
            """Wait for results, blocking until all have arrived."""
            while True:
                try:
                    self.poll(True)
                except NoResultsPending:
                    break

        def dismissWorkers(self, num_workers, do_join=False):
            """
            This is no op since the threads are already destroyed
            """
            pass

        def close(self):
            """
            Dismiss all the remaining workers
            """
            self.future_dict.clear()
            self.futures_threadpool.shutdown(wait=False)
            for request_future in concurrent.futures.as_completed(self.future_dict):
                request_future.cancel()
            self.future_dict.clear()
            self.futures_threadpool.shutdown(wait=True)

except ImportError:
    pass

################
# USAGE EXAMPLE
################

if __name__ == '__main__':
    import random
    import time

    # this will be called each time a result is available
    def print_result(threadpool, request, result):
        print "**** Result from request #%s: %r" % (request.requestID, result)

    # this will be called when an exception occurs within a thread
    # this example exception handler does little more than the default handler
    def handle_exception(threadpool, request, exc_info):
        if not isinstance(exc_info, tuple):
            # Something is seriously wrong...
            print request
            print exc_info
            raise SystemExit
        print "**** Exception occured in request #%s: %s" % \
            (request.requestID, exc_info)

    # assemble the arguments for each job to a list...
    data = [random.randint(1, 10) for i in range(20)]
    # ... and build a WorkRequest object for each item in data
    requests = makeRequests(do_something, data, print_result, handle_exception)
    # to use the default exception handler, uncomment next line and comment out
    # the preceding one.
    # requests = makeRequests(do_something, data, print_result)

    # or the other form of args_lists accepted by makeRequests: ((,), {})
    data = [((random.randint(1, 10),), {}) for i in range(20)]
    requests.extend(
        makeRequests(do_something, data, print_result, handle_exception)
        # makeRequests(do_something, data, print_result)
        # to use the default exception handler, uncomment next line and comment
        # out the preceding one.
    )

    # we create a pool of 3 worker threads
    print "Creating thread pool with 3 worker threads."
    main = ThreadPool(3, mode=THREADPOOL_FORK_MODE)

    # then we put the work requests in the queue...
    for req in requests:
        main.putRequest(req)
        print "Work request #%s added." % req.requestID
    # or shorter:
    # [main.putRequest(req) for req in requests]

    # ...and wait for the results to arrive in the result queue
    # by using ThreadPool.wait(). This would block until results for
    # all work requests have arrived:
    # main.wait()

    # instead we can poll for results while doing something else:
    i = 0
    while True:
        try:
            time.sleep(0.5)
            main.poll()
            print "Main thread working...",
            print "(active worker threads: %i)" % (threading.activeCount() - 1,)
            if i == 10:
                print "**** Adding 3 more worker threads..."
                main.createWorkers(3)
            if i == 20:
                print "**** Dismissing 2 worker threads..."
                main.dismissWorkers(2)
            i += 1
        except KeyboardInterrupt:
            print "**** Interrupted!"
            break
        except NoResultsPending:
            print "**** No pending results."
            break
    if main.dismissedWorkers:
        print "Joining all dismissed worker threads..."
        main.joinAllDismissedWorkers()
