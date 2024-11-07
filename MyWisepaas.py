import datetime
import time
import string
import random
import threading

from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, MQTTOptions, DCCSOptions, EdgeData, EdgeTag, EdgeStatus, EdgeDeviceStatus, EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig, DiscreteTagConfig, TextTagConfig
from wisepaasdatahubedgesdk.Common.Utils import RepeatedTimer


def on_connected(edgeAgent, isConnected):
  print("connected !")
  config = __generateConfig()
  _edgeAgent.uploadConfig(action = constant.ActionType['Create'], edgeConfig = config)
  
  
def on_disconnected(edgeAgent, isDisconnected):
  print("disconnected !")

def edgeAgent_on_message(agent, messageReceivedEventArgs):
  print("edgeAgent_on_message !")

def __sendData(Tag1="TagDensity", Value1=None, Tag2="Tag2", Value2=None, Tag3="Tag3", Value3=None):
  data = __generateData(Tag1, Value1, Tag2, Value2, Tag3, Value3)
  _edgeAgent.sendData(data)

def __generateData(Tag1="TagDensity", Value1=None, Tag2="Tag2", Value2=None, Tag3="Tag3", Value3=None):
  edgeData = EdgeData()
  for i in range(1, 1 + 1):
    for j in range(1, 1 + 1):
      deviceId = 'Device' + str(i)
      tagName = Tag1 # + str(j)
      value = Value1
      tag = EdgeTag(deviceId, tagName, value)
      edgeData.tagList.append(tag)

    for j in range(1, 1 + 1):
      deviceId = 'Device' + str(i)
      tagName = Tag2 # + str(j)
      value = Value2
      tag = EdgeTag(deviceId, tagName, value)
      edgeData.tagList.append(tag)

    for j in range(1, 1 + 1):
      deviceId = 'Device' + str(i)
      tagName = Tag3 # + str(j)
      value = Value3
      tag = EdgeTag(deviceId, tagName, value)
      edgeData.tagList.append(tag)
  return edgeData

def __generateConfig():
  config = EdgeConfig()
  deviceConfig = DeviceConfig(id = 'Device1',
    name = 'Device1',
    description = 'Device1',
    deviceType = 'Smart Device1',
    retentionPolicyName = '')
  
  text = TextTagConfig(name = 'TagDensity',
    description = 'TagDensity',
    readOnly = False,
    arraySize = 0)
  deviceConfig.textTagList.append(text)

  text2 = TextTagConfig(name = 'Tag2',
    description = 'Tag2',
    readOnly = False,
    arraySize = 0)
  deviceConfig.textTagList.append(text2)

  text3 = TextTagConfig(name = 'Tag3',
    description = 'Tag3',
    readOnly = False,
    arraySize = 0)
  deviceConfig.textTagList.append(text3)

  config.node.deviceList.append(deviceConfig)
  return config


_edgeAgent = None
edgeAgentOptions = EdgeAgentOptions(nodeId = '2e9a4724-1c48-48f4-a6ba-b2d8cac9e923')
edgeAgentOptions.connectType = constant.ConnectType['DCCS']
dccsOptions = DCCSOptions(apiUrl = 'https://api-dccs-ensaas.education.wise-paas.com/', credentialKey = '6ccdee100c24035e69e2d6150917aa6j')
edgeAgentOptions.DCCS = dccsOptions
_edgeAgent = EdgeAgent(edgeAgentOptions)
_edgeAgent.on_connected = on_connected
_edgeAgent.on_disconnected = on_disconnected
_edgeAgent.on_message = edgeAgent_on_message

_edgeAgent.connect()
