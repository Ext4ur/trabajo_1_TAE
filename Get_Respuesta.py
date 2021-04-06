import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

# Utilizaremos 3 archivos con datos referentes a:
# Caracterísiticas y composición del hogar
# http://microdatos.dane.gov.co/index.php/catalog/678/datafile/F133
# Datos de la vivienda
# http://microdatos.dane.gov.co/index.php/catalog/678/datafile/F137
# Educación
# http://microdatos.dane.gov.co/index.php/catalog/678/datafile/F138

# Se hace una lectura de los datos y se guarda en 3 variables distintas
df_hogar_read = \
    pd.read_csv('Caracteristicas y composicion del hogar.csv', sep=';')
df_hogar = df_hogar_read.copy()

df_hogar.drop_duplicates(subset=['DIRECTORIO', 'ORDEN'], inplace=True) # 1892

#df_vivienda = pd.read_csv('Datos de la vivienda.csv', sep=';')
#df_educacion = pd.read_csv('Educación.csv', sep=';')

# 1. Se generará la variable de salida con el número de hijos por persona
# Esta variable hará referencia al número de hijos que viven dentro del hogar
# para cada persona

# Para obtener la variable de salida utilizaremos las siguientes variables
# DIRECTORIO Identificador del hogar
# ORDEN Identificador de la persona dentro del hogar
# P6051 Relación de la persona con la cabeza del hogar
# P6081 El padre vive en este hogar? (1. Si 2. No 3. Fallecido)
# P6081S1 Número de orden del padre en caso de vivir en el hogar
# P6083 La madre vive en este hogar? (1. Si 2. No 3. Fallecido)
# P6083S1 Número de orden de la madre en caso de vivir en el hogar

# Vemos que en algunos casos la información aparece incompleta (Img1)
# con lo cual haremos 4 procesos para el número de hijos

# 1.1 Obtenemos el número de hijos de la persona que es cabeza de hogar

# En la variable P6051 encontramos la relación de la persona con la cabeza de
# hogar. Si está variable es igual a 3, entonces la persona es hija de la 
# cabeza del hogar

# Para esto, contamos el número de personas que son hijas de la cabeza de hogar
# agrupándolas por la variable DIRECTORIO, la cuál representa un hogar
df_hogar['hijos_cabeza'] = \
    df_hogar['P6051'].eq(3).groupby(df_hogar['DIRECTORIO']).transform('sum')
# Tras este proceso cada persona dentro del hogar, tendrá relacionado el número
# de hijos de la cabeza de hogar. En este trabajo, se quiere asociar esta 
# variable únicamente a la cabeza de hogar. Para esto, hacemos un filtro donde
# se busquen todas las personas que no sean cabezas de hogar y se pone la 
# columna en nulo (np.nan)
df_hogar.loc[~df_hogar['P6051'].eq(1), 'hijos_cabeza'] = np.nan

# 1.2 Obtenemos los padres de las cabezas de hogar que viven dentro del
# mismo hogar 
df_hogar.loc[df_hogar['P6051'].eq(5), 'padres_cabeza'] = 1

# 1.3 Obtenemos las personas que tienen hijos dentro del hogar pero que no son
# ni la cabeza del hogar ni los padres de la cabeza del hogar

# 1.3.1 Creamos 2 nuevas columnas, una indicando que la persona tiene el padre
# dentro del mismo hogar y otra indicando que la persona tiene la madre dentro
# del mismo hogar

df_hogar.loc[df_hogar['P6081'].eq(1), 'hijos_extra_padre'] = 1
df_hogar.loc[df_hogar['P6083'].eq(1), 'hijos_extra_madre'] = 1

# 1.3.2 Se crean 2 DataFrame temporales 
df_hogar_temp_padre = df_hogar[['DIRECTORIO', 'P6081S1', 'hijos_extra_padre']]
df_hogar_temp_madre = df_hogar[['DIRECTORIO', 'P6083S1', 'hijos_extra_madre']]

# 1.3.3 Se reescriben los DataFrame temporales por la suma de hijos de cada
# persona, esto colocará las columnas DIRECTORIO y P608_S1 como índices
df_hogar_temp_padre = \
    df_hogar_temp_padre.groupby(['DIRECTORIO', 'P6081S1'])\
        .sum('hijos_extra_padre')
df_hogar_temp_madre = \
    df_hogar_temp_madre.groupby(['DIRECTORIO', 'P6083S1'])\
        .sum('hijos_extra_madre')

# 1.3.4 En el DataFrame inicial, reemplazamos los espaciones en blanco de las
# celdas por un valor nulo, esto con el fin de manejar más fácilmente los datos
df_hogar.replace(' ', np.nan, inplace=True)
df_hogar.replace('', np.nan, inplace=True)

# 1.3.5 Convertimos las columnas P6081S1 y P6083S1, referentes al número de
# orden del padre y la madre respectivamente, de tipo string (por los espacios
# en blanco que tenía) a tipo flotante (np.nan se puede tomar como flotante)
df_hogar = df_hogar.astype({'P6081S1': float, 'P6083S1': float})

# 1.3.6 Reiniciamos los índices de los DataFrame temporales para trabajarlos
# como columnas
df_hogar_temp_padre.reset_index(inplace=True)
df_hogar_temp_madre.reset_index(inplace=True)

# 1.3.7 En los DataFrames temporales reemplazamos los espacios en blanco por
# valores nulos y convertimos las columnas a tipo flotante
df_hogar_temp_padre.replace(' ', np.nan, inplace=True)
df_hogar_temp_padre.replace('', np.nan, inplace=True)
df_hogar_temp_padre = df_hogar_temp_padre.astype(float)

df_hogar_temp_madre.replace(' ', np.nan, inplace=True)
df_hogar_temp_madre.replace('', np.nan, inplace=True)
df_hogar_temp_madre = df_hogar_temp_madre.astype(float)

# 1.3.8 Del DataFrame inicial se eliminan las columnas P6081S1, P6083S1,
# hijos_extra_madre e hijos_extra_padre, esto con el fin de no tener columnas
# repetidas
df_hogar.drop(columns=['P6081S1', 'P6083S1', 'hijos_extra_madre', \
                        'hijos_extra_padre'], inplace=True)

# 1.3.9 Unimos los DataFrame temporales con el DataFrame inicial por medio del
# DIRECTORIO y el ORDEN de la persona (En los DataFrame temporales, esta columna
# es P608_S1)
df_hogar = pd.merge(df_hogar, df_hogar_temp_padre,
                    left_on=['DIRECTORIO', 'ORDEN'],
                    right_on=['DIRECTORIO', 'P6081S1'], how='left')
df_hogar = pd.merge(df_hogar, df_hogar_temp_madre,
                    left_on=['DIRECTORIO', 'ORDEN'],
                    right_on=['DIRECTORIO', 'P6083S1'], how='left')

# 1.4 Unimos los datos del número de hijos de todas las columnas generadas

# 1.4.1 Se crea una columna hijos_extra, donde se sumarán las siguientes
# columnas:
# - hijos_extra_padre: Número de hijos de quienes son padres dentro del hogar
# pero no son la cabeza de hogar
# - hijos_extra_madre: Número de hijos de quienes son madres dentro del hogar
# pero no son la cabeza de hogar

df_hogar['hijos_extra'] = \
    df_hogar[['hijos_extra_padre', 'hijos_extra_madre']].sum(axis=1)

# Verificamos si las columnas tienen datos en una misma fila
#! print(df_hogar.loc[(df_hogar['hijos_extra_padre'].notna() &
#!                     df_hogar['hijos_extra_padre'].ne(0)) &
#!                     (df_hogar['hijos_extra_madre'].notna() &
#!                     df_hogar['hijos_extra_madre'].ne(0))])

# 1.4.2 Creamos una columna hijos_padre_cabeza, donde se guardarán los hijos de
# los padres de las cabezas de hogar, esto se hará ya que algunos datos no 
# coincident entre las columnas padres_cabeza e hijos_extra (Img2) creada
# anteriormente

df_hogar.loc[df_hogar['padres_cabeza'].notna() &
                df_hogar['padres_cabeza'].lt(df_hogar['hijos_extra']),
                'hijos_padre_cabeza'] = df_hogar['hijos_extra']

df_hogar.loc[df_hogar['padres_cabeza'].notna() &
                df_hogar['padres_cabeza'].gt(df_hogar['hijos_extra']),
                'hijos_padre_cabeza'] = df_hogar['padres_cabeza']

df_hogar.loc[df_hogar['padres_cabeza'].notna() &
                df_hogar['padres_cabeza'].eq(df_hogar['hijos_extra']),
                'hijos_padre_cabeza'] = df_hogar['padres_cabeza']

df_hogar.loc[df_hogar['padres_cabeza'].notna() &
                df_hogar['hijos_extra'].isna(),
                'hijos_padre_cabeza'] = df_hogar['padres_cabeza']

df_temp = \
df_hogar.loc[df_hogar['padres_cabeza'].notna() &
                df_hogar['hijos_padre_cabeza'].isna()]

# Verificamos que el número de hijos de los padres de las personas cabezas de
# hogar esté correcto (No tenga datos nulos)
#! print(df_temp[['DIRECTORIO', 'ORDEN', 'padres_cabeza', 'hijos_extra',
#!                 'hijos_padre_cabeza']])

# 1.4.3 Creamos una columna hijos_cabeza_check, donde se guardará el número de
# hijos de la cabeza del hogar, verificando tanto en la columna hijos_cabeza
# como en la columna hijos_extra

df_hogar.loc[df_hogar['hijos_cabeza'].notna() &
                df_hogar['hijos_cabeza'].gt(df_hogar['hijos_extra']),
                'hijos_cabeza_check'] = df_hogar['hijos_cabeza']

df_hogar.loc[df_hogar['hijos_cabeza'].notna() &
                df_hogar['hijos_cabeza'].lt(df_hogar['hijos_extra']),
                'hijos_cabeza_check'] = df_hogar['hijos_cabeza']

df_hogar.loc[df_hogar['hijos_cabeza'].notna() &
                df_hogar['hijos_cabeza'].eq(df_hogar['hijos_extra']),
                'hijos_cabeza_check'] = df_hogar['hijos_cabeza']

# Verificamos si hay algún dato faltante
#! print(df_hogar[['hijos_cabeza', 'hijos_extra']]\
#!         .loc[df_hogar['hijos_cabeza'].notna() &
#!                 df_hogar['hijos_cabeza'].ne(df_hogar['hijos_extra']) &
#!                 df_hogar['hijos_cabeza_check'].isna()])


# 1.4.5 Se crea una columna hijos, donde se guardarán el número de hijos de cada
# persona en base a las siguientes columnas:
# - hijos_cabeza_check: Número de hijos de la cabeza del hogar
# - hijos_padre_cabeza: Número de hijos de los padres de la cabeza del hogar
# - hijos_extra: Hijos de quienes no son cabeza de hogar

# Verificamos que ninguna fila tenga datos de las cabezas de hogar y de sus
# padres (Esto sería un error de los datos donde un hijo es padre de su padre o
# también puede verse como una paradoja espacio temporal)

#! df_temp = \
#!     df_hogar.loc[df_hogar['hijos_cabeza_check'].notna() &
#!                     df_hogar['hijos_padre_cabeza'].notna()]
#! 
#! print(df_temp)

# 1.4.6 Guardamos el número de hijos de las cabezas de hogar
df_hogar.loc[df_hogar['hijos_cabeza_check'].notna(), 'hijos'] = \
    df_hogar['hijos_cabeza_check']

#! print('Número de hijos con datos de las cabezas de hogar',
#!         len(df_hogar.loc[df_hogar['hijos'].notna()])) # 93161

# 1.4.7 Guardamos el número de hijos de los padres de las cabezas de hogar
df_hogar.loc[df_hogar['hijos_padre_cabeza'].notna(), 'hijos'] = \
    df_hogar['hijos_padre_cabeza']

#! print('Número de hijos con datos de las cabezas de hogar y sus padres',
#!         len(df_hogar.loc[df_hogar['hijos'].notna()])) # 96264

# 1.4.8 Guardamos el número de hijos de los demás habitantes del hogar
df_hogar.loc[df_hogar['hijos_cabeza_check'].isna() &
                df_hogar['hijos_padre_cabeza'].isna() &
                df_hogar['hijos_extra'], 'hijos'] = \
                df_hogar['hijos_extra']

#! print('Número de hijos con datos de las cabezas de hogar, sus padres y los '
#!         'demás habitantes del hogar',
#!         len(df_hogar.loc[df_hogar['hijos'].notna()])) # 145898

# 1.5 Eliminamos las columnas auxiliares utilizadas

#! print(df_hogar.columns)
df_hogar\
    .drop(columns=['hijos_cabeza', 'padres_cabeza', 'hijos_extra_padre',
                    'hijos_extra_madre', 'hijos_extra', 'hijos_padre_cabeza',
                    'hijos_cabeza_check'], inplace=True)
#! print(df_hogar.columns)

# 2. Se unen los datos de los otros dos archivos que se utilizarán con los datos
# de la composición del hogar

# 2.1 Lectura de los archivos faltantes

# 2.1.1 Guardamos las características de la vivienda, este archivo contiene un
# único dato por hogar y no uno por cada habitante del hogar
df_vivienda_read = pd.read_csv('Datos de la vivienda.csv', sep=';')
df_vivienda = df_vivienda_read.copy()

#df_vivienda = pd.read_csv('Datos de la vivienda.csv', sep=';')
#df_educacion = pd.read_csv('Educación.csv', sep=';')

# 2.1.1 Guardamos la educación de las personas, este archivo contiene un dato
# por cada persona, por lo que se eliminarán los duplicados filtrando por el
# DIRECTORIO y el ORDEN
df_educacion_read = pd.read_csv('Educación.csv', sep=';')
df_educacion = df_educacion_read.copy()

df_educacion.drop_duplicates(subset=['DIRECTORIO', 'ORDEN'], inplace=True)

# Vemos las longitudes de los datos
print(len(df_hogar))
print(len(df_vivienda))
print(len(df_educacion))

# Notamos que la longitud de los datos sobre la educación es menor que la de los
# datos del hogar, sin embargo ambos deberían tener la misma longitud, se
# tomarán los datos de educación, con lo que algunos datos tendrán esta
# información nula

# Unimos los 3 DataFrames en uno solo
df_vivienda.drop(columns=['SECUENCIA_ENCUESTA', 'SECUENCIA_P', 'ORDEN',
                            'FEX_C'], inplace=True)
df = pd.merge(df_hogar, df_vivienda, on=['DIRECTORIO'])

df = pd.merge(df, df_educacion, on=['DIRECTORIO', 'SECUENCIA_ENCUESTA',
                'SECUENCIA_P', 'ORDEN', 'FEX_C'], how='left')

# 3. Elección de las columnas a estudiar
# Para este estudio tomaremos en cuenta las siguientes columnas
# DIRECTORIO Id_hogar
# ORDEN orden dentro del directorio
# P6020 Sexo (Hombre, Mujer)
# P6040 Edad
# P5502 Sentimental (Pareja, Viudo, Separado, Soltero, Casado)
# P756 Nacimiento (Este municipio, Otro municipio, país)
# P6080 Cultura
# P2057 Se considera campesino
# P1895 Sastisfacción con vida
# P1896 Satisfacción con ingreso
# P1897 Satisfacción con salud
# P1898 Satisfacción con seguridad
# P1899 Satisfacción con trabajo

# P1070 Tipo vivienda (Casa, Apartamento, Cuarto, Indígena, Otro)
# P8520S1 Energía eléctrica (Si, No)
# P8520S5 Acueducto
# P8520S3 Alcantarillado
# P8520S4 Recolección de basuras

# P6160 Sabe leer y escribir (Si, No)
# P8587 Nivel educativo más alto alcanzado

# Algunas de estas columnas serán manipuladas para sintetizar los datos

# 3.1 Síntesis de datos
# P5502 Sentimental (Pareja, Viudo, Separado, Soltero, Casado)
# Se creará una columna binaria relacion, si se encuentra casado (actualmente
# casado) o no (Pareja, Viudo, Separado, Soltero)
# 1 Si no está casado, 2 si está casado

df['relacion'] = df['P5502'].astype(float)

# P1895 Sastisfacción con vida
# P1896 Satisfacción con ingreso
# P1897 Satisfacción con salud
# P1898 Satisfacción con seguridad
# P1899 Satisfacción con trabajo
# Se agruparán en una única columna satisfaccion referente al promedio entre
# estas

df = df.astype({'P1895': float, 'P1896': float, 'P1897': float, 'P1898': float,
                'P1899': float})

df.loc[df['P1896'].eq(99), 'P1896'] = 9

df['satisfaccion'] = \
    df[['P1895', 'P1896', 'P1897', 'P1898', 'P1899']].mean(axis=1)

# P1070 Tipo vivienda (Casa, Apartamento, Cuarto, Indígena, Otro)
# Se creará una columna vivienda, la cuál tomará Casa, Apartamento u Otro
# (Agrupando Cuarto, Indígena y otro)
# 1 para Casa, 2 para Apartamento, 3 para Otro
df.loc[df['P1070'].eq(1), 'vivienda'] = 1
df.loc[df['P1070'].eq(2), 'vivienda'] = 2
df.loc[df['P1070'].isin([3, 4, 5]), 'vivienda'] = 3

df = df.astype({'P2057': float})
df.loc[df['P2057'].isin([np.nan, 9]), 'P2057'] = 3

df['P8587'].replace(' ', np.nan, inplace=True)
df['P8587'].replace('', np.nan, inplace=True)
df = df.astype({'P8587': float})

# Se eliminan las columnas anteriores
df.drop(columns=['P5502', 'P1895', 'P1896', 'P1897', 'P1898', 'P1899', 'P1070'],
        inplace=True)

# Creamos una archivo excel con las columnas a estudiar, con este archivo
# trabajaremos el modelo de regresión
df1 = df.loc[df['hijos'].notna()]

df1.replace(np.nan, 0, inplace=True)

df1[['DIRECTORIO', 'ORDEN', 'P6020', 'P6040', 'relacion', 'P756', 'P6080',
    'P2057', 'satisfaccion', 'vivienda', 'P8520S1', 'P8520S5', 'P8520S3',
    'P8520S4', 'P6160', 'P8587', 'hijos']].to_excel('Datos.xlsx', index=False)