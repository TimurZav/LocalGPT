import logging
from logging.handlers import RotatingFileHandler


class SingleLineLogHandler(RotatingFileHandler):
    def emit(self, record):
        msg = self.format(record)
        if self.stream is not None:
            self.stream.write(f'{msg}')
            self.stream.flush()


class FileLogger(logging.Logger):
    def __init__(
        self,
        name,
        filename,
        mode='a',
        level=logging.INFO,
        fformatter=None,
        log_to_console=False,
        sformatter=None
    ):
        super().__init__(name, level)

        # Create a custom file handler
        self.file_handler = SingleLineLogHandler(filename=filename, mode=mode, maxBytes=1.5 * pow(1024, 2),
                                                 backupCount=3)

        # Set the formatter for the file handler
        if fformatter is not None:
            self.file_handler.setFormatter(logging.Formatter('%(message)s'))

        # Add the file handler to the logger
        self.addHandler(self.file_handler)

        if log_to_console:
            # Create a console handler
            self.console_handler = logging.StreamHandler()  # Prints to the console

            # Set the formatter for the console handler
            if not sformatter:
                sformatter = fformatter
            self.console_handler.setFormatter(sformatter)

            # Add the console handler to the logger
            self.addHandler(self.console_handler)

    def fdebug(self, msg, pre_msg=''):
        if pre_msg:
            print(pre_msg)
        self.debug(msg)

    def finfo(self, msg):
        self.info(msg)

    def fwarn(self, msg):
        self.warning(msg)

    def ferror(self, msg):
        self.error(msg)

    def fcritical(self, msg):
        self.critical(msg)


# Test the logging
if __name__ == '__main__':
    log_format = '%(message)s'
    logging.basicConfig(format=log_format, level=logging.CRITICAL, datefmt="%H:%M:%S")
    fLogger = FileLogger(__name__, 'tmp.log', mode='a', level=logging.INFO)
    fLogger.finfo("le")
    fLogger.finfo("vel")

