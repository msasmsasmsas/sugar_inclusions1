# coding=utf-8
import os
import logging
import time
from tqdm import tqdm
from datetime import datetime
from ctypes import *

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Enum import EM_USEDEV_MODE, EM_QUERY_RECORD_TYPE
from NetSDK.SDK_Struct import NET_RECORDFILE_INFO, C_DWORD
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect

from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE
from NetSDK.SDK_Struct import NET_TIME, C_LDWORD, C_LLONG, NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY, NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY

logging.basicConfig(level=logging.INFO)

def DisConnectCallBack(lLoginID, pchDVRIP, nDVRPort, dwUser):
    print("PlayBack OffLine")
def ReConnectCallBack(lLoginID, pchDVRIP, nDVRPort, dwUser):
    print("PlayBack OnLine")

@WINFUNCTYPE(None, C_LLONG, C_DWORD, C_DWORD, C_LDWORD)
def DownLoadPosCallBack(lLoginID, pchDVRIP, nDVRPort, dwUser):
    pass

@WINFUNCTYPE(c_int, C_LLONG, C_DWORD, POINTER(c_ubyte), C_DWORD, C_LDWORD)
def DownLoadDataCallBack(lPlayHandle, dwDataType, pBuffer, dwBufSize, dwUser):
    # buf_data = cast(pBuffer, POINTER(c_ubyte * dwBufSize)).contents
    # with open('./buffer.dav', 'ab+') as buf_file:
    #     buf_file.write(buf_data)
    return 1

@WINFUNCTYPE(None, C_LLONG, C_DWORD, C_DWORD, c_int, NET_RECORDFILE_INFO, C_LDWORD)
def TimeDownLoadPosCallBack(lPlayHandle, dwTotalSize, dwDownLoadSize, index, recordfileinfo, dwUser):
    try:
        dnd.update_download_progress(dwTotalSize,dwDownLoadSize)
    except Exception as e:
        logging.error(e)

class DahuaNVRDownloader(object):
    def __init__(self,ip,user,password,stream_type):
        self.ip = ip
        self.user = user
        self.password = password
        self.stream_type = stream_type

        self.sdk = NetClient()
        m_DisConnectCallBack = fDisConnect(DisConnectCallBack)
        m_ReConnectCallBack = fHaveReConnect(ReConnectCallBack)
        self.sdk.InitEx(m_DisConnectCallBack)
        self.sdk.SetAutoReconnect(m_ReConnectCallBack)

    def update_download_progress(self,total_size,download_size):
        try:
            logging.debug("total_size:{} download_size:{}".format(total_size, download_size))
            if download_size == -1:
                self.sdk.StopDownload(self.downloadID)
                logging.warning("Download End")
            elif download_size == -2:
                logging.error("Download Failed")
                self.StopDownload()
            else:
                if download_size >= total_size:
                    self.progress_bar.update(total_size*1024 - self.progress_bar.n)

                    self.StopDownload()
                else:
                    #print("")
                    if self.progress_bar is None:
                       self.progress_bar = tqdm(total=total_size*1024, unit='B', unit_scale=True)
                    else:
                       self.progress_bar.update(download_size*1024-self.progress_bar.n)
        except Exception as e:
            logging.error(e)


    def StopDownload(self):
        self.processing = False
        #print('Stop Download')

    def Login(self):

        stuInParam = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuInParam.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        stuInParam.szIP = self.ip.encode()
        stuInParam.nPort = 37777
        stuInParam.szUserName = self.user.encode()
        stuInParam.szPassword = self.password.encode()
        stuInParam.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        stuInParam.pCapParam = None

        stuOutParam = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuOutParam.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

        self.loginID, self.device_info, self.error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        login_result = self.loginID != 0
        if login_result:
            logging.info(self.device_info)
            logging.info("login_ok" + self.error_msg)
            self.set_stream_type(self.stream_type)
        else:
            logging.error(str(self.sdk.GetLastError()))
        return login_result

    def set_stream_type(self,stream_type):
        stream_type = c_int(stream_type)
        result = self.sdk.SetDeviceMode(self.loginID, int(EM_USEDEV_MODE.RECORD_STREAM_TYPE), stream_type)
        if not result:
            logging.error(str(self.sdk.GetLastError()))
            return 0, 0, None

    def ConvertStrToDateTime(self,time_start,time_end):
        date_time_start = datetime.strptime(time_start, '%d-%m-%Y %H:%M:%S')
        date_time_end = datetime.strptime(time_end, '%d-%m-%Y %H:%M:%S')

        startDateTime = NET_TIME()
        startDateTime.dwYear = date_time_start.year
        startDateTime.dwMonth = date_time_start.month
        startDateTime.dwDay = date_time_start.day
        startDateTime.dwHour = date_time_start.hour
        startDateTime.dwMinute = date_time_start.minute
        startDateTime.dwSecond = date_time_start.second

        enddateTime = NET_TIME()
        enddateTime.dwYear = date_time_end.year
        enddateTime.dwMonth = date_time_end.month
        enddateTime.dwDay = date_time_end.day
        enddateTime.dwHour = date_time_end.hour
        enddateTime.dwMinute = date_time_end.minute
        enddateTime.dwSecond = date_time_end.second
        return startDateTime,enddateTime

    def DownloadByFile(self, save_file_path, nchannel, time_start, time_end):
        startDateTime, enddateTime = self.ConvertStrToDateTime(time_start, time_end)
        result, fileCount, infos = self.sdk.QueryRecordFile(self.loginID, nchannel, int(EM_QUERY_RECORD_TYPE.ALL), startDateTime,
                                                            enddateTime, None, 5000, False)
        if result:
            logging.info("fileCount: {} ".format(fileCount))
            if len(infos) < fileCount:
               fileCount = len(infos)
            if fileCount > 0:
                for idx in range(fileCount):
                    info = infos[idx]
                    #base_name = str(os.path.basename(info.filename.decode('ascii')))
                    st = info.starttime
                    et = info.endtime

                    file_name = "{:02d}_{:02d}_{}_{:02d}_{:02d}_{:02d}-".format(st.dwDay,st.dwMonth,st.dwYear,st.dwHour,st.dwMinute,st.dwSecond)+\
                                "{:02d}_{:02d}_{}_{:02d}_{:02d}_{:02d}.dav".format(et.dwDay,et.dwMonth,et.dwYear,et.dwHour,et.dwMinute,et.dwSecond)

                    save_file_name = os.path.join(save_file_path,file_name)
                    logging.info("Downloading file: {}/{} {}  Size:{} Mb".format(idx+1, fileCount, save_file_name, info.size/1024))
                    self.DoDownload(save_file_name,nchannel,info.starttime,info.endtime,info.size)
        else:
            logging.error(str(self.sdk.GetLastError()))


    def DownloadByTime(self, save_file_name, nchannel, time_start, time_end):
        logging.info("Sarting download:{} - {}".format(time_start, time_end, nchannel))
        startDateTime, enddateTime = self.ConvertStrToDateTime(time_start, time_end)
        self.DoDownload(save_file_name,nchannel,startDateTime,enddateTime)

    def DoDownload(self,save_file_name,nchannel,startDateTime,enddateTime,total_size=0):
        if total_size == 0:
            self.progress_bar = None
        else:
            self.progress_bar = tqdm(total=total_size * 1024, unit='B', unit_scale=True)
        self.downloadID = self.sdk.DownloadByTimeEx(self.loginID, nchannel, int(EM_QUERY_RECORD_TYPE.ALL),
                                                    startDateTime, enddateTime, save_file_name,
                                                    TimeDownLoadPosCallBack, 0,
                                                    DownLoadDataCallBack, 0)
        self.processing = True
        while self.processing:
            time.sleep(0.1)
        self.sdk.StopDownload(self.downloadID)
        self.progress_bar.close()


if __name__ == '__main__':

    dnd = DahuaNVRDownloader("192.168.0.108","admin","AstartA_$ec",0) #0 - основной канал, 1 - дополнительный
    if dnd.Login():
        #dnd.DownloadByTime(r"d:\temp\out.dav",2,"15-10-2020 09:00:00","15-10-2020 09:10:00") #r"d:\temp\out.dav" - ���� � ������� �������� �������
         dnd.DownloadByFile(r"E:\TEMP_OCT", 2, "28-10-2020 08:00:00", "28-10-2020 17:20:00")  #"E:\TEMP_OCT" - ���� ������, #0 - ����� ������ , ���� � ������� DD-MM-YYYY HH:MM:SS

