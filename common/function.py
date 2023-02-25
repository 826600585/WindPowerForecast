import os
import configparser

def getConfig(filename,section,option):
    proDir = os.path.split(os.path.realpath(__file__))[0]
    configPath = os.path.join(proDir,filename)
    conf = configparser.ConfigParser()
    conf.read(configPath)
    config = conf.get(section,option)
    return config