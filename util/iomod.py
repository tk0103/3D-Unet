# coding: utf-8

import sys
import os
import numpy as np


def load_raw(path, dtype):
    """
    LOAD PLANE BINARY DATA
    raw画像を読み込んでreshapeする関数
    path:input file path
    dtype:type of data
        np.int8        --:chara
        np.uint8       --:uchara
        np.int16       --:short
        np.int32       --:int
        np.uint32      --:long
        np.float32     --:float
        np.float64     --:double
    made by tkt ,2013/11/
    """

    argvs  = sys.argv	#引数を格納したリストの取得
    argc = len(argvs)	#引数の数を確認

    """
    if (argc != 1):		#引数の合はエラー
        print "The number of input is incorrect !"
        quit()		#プログラム終了
    """

    if (dtype == 'short') or (dtype == np.int16):
        type = np.int16
    elif (dtype == 'int') or (dtype == np.int32):
        type = np.int32
    elif (dtype == 'ushort') or (dtype == np.uint16):
        type = np.uint16
    elif (dtype == 'uint') or (dtype == np.uint32):
        type = np.uint32
    elif (dtype == 'float') or (dtype == np.float32):
        type = np.float32
    elif (dtype == 'double') or (dtype == np.float64):
        type = np.float64
    elif (dtype == 'char') or (dtype == np.int8):
        type = np.int8
    elif (dtype == 'uchar') or (dtype == np.uint8):
        type = np.uint8
    else:
        print(korakora)
        quit()

    data = np.fromfile(path, type)

    return data


def load_raw_and_mhd( mhd_path ):
    """
    IMPORT 3D CT IMAGES WITH MHD HEADER
    """
    dirname = os.path.dirname(mhd_path)
    fid = open(mhd_path, 'rt')
    C = fid.read()
    V = C.split("\n")
    V.pop()

    for i in range(0, len(V)):
        VV = V[i].split("=")
        if "NDims" in VV[0]:
            dim = VV[1]
            dim = dim.strip()
            dim = int(dim)

        elif "DimSize" in VV[0]:
            dimsize = VV[1]
            dimsize = dimsize.strip()
            dimsize = dimsize.split(' ')
            dimsize = map(int,dimsize)

        elif "ElementType" in VV[0]:
            dtype = VV[1]
            dtype = dtype.strip()

        elif "ElementDataFile" in VV[0]:
            filename = VV[1]
            filename = filename.strip()

        else:
            pass

    C = V

    if dtype == "MET_CHAR":
        dtype = np.int8
    elif dtype == "MET_UCHAR":
        dtype = np.uint8
    elif dtype == "MET_SHORT":
        dtype = np.int16
    elif dtype == "MET_USHORT":
        dtype = np.uint16
    elif dtype == "MET_FLOAT":
        dtype = np.float32
    elif dtype == "MET_DOUBLE":
        dtype = np.float64
    elif dtype == "MET_INT":
        dtype = np.uint32
    else:
        print(korakora)
        quit()		#プログラム終了

    path = os.path.join(dirname, filename)
    #print path
    I = load_raw(path, dtype)

    """
    dimsize.reverse() # 岸本追加 (x,y,z) → (z,y,x)
    I = I.reshape(dimsize)
    """

    return C, I


def save_raw(Data, path, dtype):
    """
        バイナリデータの保存
        path : 入力ファイルのパス
        dtype:
        'short'     np.int16
        'int'       np.int32
        'ushort'    np.uint16
        'uint'      np.uint32
        'float'     np.float32
        'double'    np.float64
        'char'      np.int8
        'uchar'     np.uint8
    """

    data_dir, file_name = os.path.split(path)

    if os.path.isdir(data_dir):
        pass
    else:
        os.mkdir(data_dir)

    if dtype == 'short':
        Data.astype(np.int16)
    elif dtype == 'int':
        Data.astype(np.int32)
    elif dtype == 'ushort':
        Data.astype(np.uint16)
    elif dtype == 'uint':
        Data.astype(np.uint32)
    elif dtype == 'float':
        Data.astype(np.float32)
    elif dtype == 'double':
        Data.astype(np.float64)
    elif dtype == 'char':
        Data.astype(np.int8)
    elif dtype == 'uchar':
        Data.astype(np.uint8)
    else:
       pass

    fid = open(path, 'wb')
    fid.write(Data)
    fid.close()

def save_raw_and_mhd(C, I, raw_path, dtype):

    data_dir, file_name = os.path.split(raw_path)
    print(data_dir)
    print(file_name)
    name, ext = os.path.splitext(file_name)
    mhd_name = name + '.mhd'
    mhd_path = data_dir + "\\" + mhd_name

    if os.path.isdir(data_dir):
        pass
    else:
        os.mkdir(data_dir)

    for i in range(0,len(C)):
        VV = C[i].split("=")

        if "ObjectType" in VV[0]:
            otype = VV[1]
            otype = otype.strip()
            print("otype =", otype)

        elif "NDims" in VV[0]:
            Ndims = VV[1]
            Ndims = Ndims.strip()
            print(Ndims)

        elif "ElementSpacing" in VV[0]:
            ElementSpacing = VV[1]
            ElementSpacing = ElementSpacing.strip()
            print(ElementSpacing)
        else:
            pass

    if (dtype == 'short') or (dtype == np.int16):
        dtype = np.int16
        type_name = 'MET_SHORT'
    elif (dtype == 'int') or (dtype == np.int32):
        dtype = np.int32
        type_name = 'MET_INT'
    elif (dtype == 'ushort') or (dtype == np.uint16):
        dtype = np.uint16
        type_name = 'MET_USHORT'
    elif (dtype == 'uint') or (dtype == np.uint32):
        dtype = np.uint32
        type_name = 'MET_UINT'
    elif (dtype == 'float') or (dtype == np.float32):
        dtype = np.float32
        type_name = 'MET_FLOAT'
    elif (dtype == 'double') or (dtype == np.float64):
        dtype = np.float64
        type_name = 'MET_DOUBLE'
    elif (dtype == 'char') or (dtype == np.int8):
        dtype = np.int8
        type_name = 'MET_CHAR'
    elif (dtype == 'uchar') or (dtype == np.uint8):
        dtype = np.uint8
        type_name = 'MET_UCHAR'
    else:
        print(korakora)

    DimSize = I.shape
    DimSize = map(str,DimSize)
    DimSize = ",".join(DimSize)
    DimSize = DimSize.replace(',', ' ')

    STR = ["ObjectType = " + otype, '\n', "NDims = " + Ndims, '\n', "DimSize = " + DimSize, '\n', "ElementSpacing = " + ElementSpacing, '\n', "ElementType = " + type_name, '\n', "ElementDataFile = " + file_name]

    fid = open(mhd_path, 'w')
    fid.writelines(STR)
    fid.close()

    save_raw(I, raw_path, dtype)