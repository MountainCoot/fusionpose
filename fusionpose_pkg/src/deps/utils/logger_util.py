try:
    import logging
    import os
    import rospkg
    import time

    logging_setup = False

    class CustomHandler(logging.Handler):
        def __init__(self, filename: str) -> None:
            super().__init__()
            self.filename = filename

        def emit(self, record: logging.LogRecord) -> None:
            try:
                with open(self.filename, 'a') as file:
                    file.write(self.format(record) + '\n')
                    file.flush()
            except Exception as e:
                print(f"Error writing log to {self.filename}: {e}")
                print(f"Log record: {record}")

    def reset_logging(name='root', log_file_dir=None, log_levels=None, replay=False) -> logging.Logger:
        if log_levels is None:
            log_levels = [['all']]  # Default to logging everything in one file

        if log_file_dir is None:
            log_file_dir = os.path.join(os.getcwd(), 'logs')

        os.makedirs(log_file_dir, exist_ok=True)

        # Clear old log files if modified too long ago
        global logging_setup
        if not logging_setup:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            
            # Create a handler for each specified log level combination
            for levels in log_levels:
                # Set log file name based on levels
                file_suffix = '_'.join(level.lower() for level in levels)
                if replay:
                    log_file_path = os.path.join(log_file_dir, f'logging_replay_{file_suffix}.log')
                else:
                    log_file_path = os.path.join(log_file_dir, f'logging_{file_suffix}.log')
                
                # Check and clear file if outdated
                if os.path.exists(log_file_path) and (time.time() - os.path.getmtime(log_file_path)) > 3:
                    open(log_file_path, 'w').close()
                elif not os.path.exists(log_file_path):
                    open(log_file_path, 'w').close()
                
                # Set minimum level for this handler
                if 'all' in levels:
                    handler_level = logging.DEBUG
                else:
                    # Use the lowest level in the specified levels
                    level_map = {
                        'debug': logging.DEBUG, 'info': logging.INFO,
                        'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL
                    }
                    handler_level = min(level_map[level] for level in levels)
                
                handler = CustomHandler(log_file_path)
                handler.setLevel(handler_level)
                handler.setFormatter(logging.Formatter(
                    '{asctime}  {levelname:5s} \t{message} [{filename}:{lineno}]', 
                    style='{', datefmt='%H:%M:%S'
                ))

                # Add filtering logic to only log specific levels in this handler
                handler.addFilter(lambda record, lvls=levels: 
                                'all' in lvls or record.levelname.lower() in lvls)
                
                logger.addHandler(handler)

            print("Logging directory: ", os.path.abspath(os.path.dirname(log_file_path)))
            
            # Prevent logs from being printed to console
            logger.propagate = False
            logging_setup = True
            return logger
        else:
            return logging.getLogger(name)
        
    try:
        pkg_name = 'fusionpose_pkg'
        rospack_path = rospkg.RosPack().get_path(pkg_name)
        # log_file_idr = os.path.join(rospack_path, 'logs', 'live',
        log_file_dir = os.path.join(rospack_path, 'logs', 'live')
        log_levels = []
        log_levels.append(['all'])
        log_levels.append(['info', 'warning', 'error'])
        log_levels.append(['critical'])
        log_levels.append(['debug'])
        Logger = reset_logging(pkg_name, log_file_dir, log_levels=log_levels)
    except:
        log_file_dir = os.path.join(os.getcwd(), 'logs', 'replay')
        Logger = reset_logging('my_logger', log_file_dir, replay=True)
        
    def get_path():
        return log_file_dir
    

    def reset_external():
        reset_logging(pkg_name, log_file_dir)

except ModuleNotFoundError:
    # remap logger to print if logger_util is not available
    class Logger:
        @staticmethod
        def info(msg: str) -> None:
            print(msg)
        @staticmethod
        def warning(msg: str) -> None:
            print(msg)
        @staticmethod
        def error(msg: str) -> None:
            print(msg)
        @staticmethod
        def debug(msg: str) -> None:
            print(msg)
        @staticmethod
        def exception(msg: str) -> None:
            print(msg)
        @staticmethod
        def critical(msg: str) -> None:
            print(msg)
        