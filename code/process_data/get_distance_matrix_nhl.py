from geopy.geocoders import Nominatim
import geopy.distance
import os

import pandas as pd

geolocator = Nominatim(user_agent="my_app")




def calculate_geo_position(df):
    # Funcion que, dada un string de direccion, calcula la latitud y longitud de un punto
    # Pasamos la direccion a un GeoCode y luego a coordenadas
    df['Direccion'] = df['Arena'].apply(geolocator.geocode)
    df['Coordenadas'] = df['Direccion'].apply(lambda x: (x.latitude, x.longitude))
    df['Latitud'] = df['Direccion'].apply(lambda x: x.latitude)
    df['Longitud'] = df['Direccion'].apply(lambda x: x.longitude)
    return df


def distance_calculator(df):
    # Funcion que crea una matriz de distancia entre todos los clubes

    # Lista de todos los clubes
    clubes = list(df["Team"])

    # Por otro lado, generamos un diccionario donde guardaremos las distintas distancias para cada combinacion de clubes
    distances = {}

    # Vamos calculando las distancias
    for i in range(len(clubes)):
        distances[clubes[i]] = []
        for j in range(len(clubes)):
            distance = geopy.distance.distance(df["Coordenadas"][i], df["Coordenadas"][j]).km
            distances[clubes[i]].append(distance)

    # Generamos el data set con la matriz de distancia
    distance_matrix = pd.DataFrame()
    distance_matrix["Equipo"] = clubes

    for i in range(len(list(distances.keys()))):
        distance_matrix[list(distances.keys())[i]] = distances[list(distances.keys())[i]]

    return distance_matrix


if __name__ == '__main__':
    cmd = os.getcwd()
    cmd = cmd.replace('code\\process_data', 'data\\teams\\nhl')
    df = pd.read_csv(f'{cmd}/nhl_locations.csv', sep=';')

    df = calculate_geo_position(df)
    df
    matriz = distance_calculator(df)
    matriz.to_csv(f'{cmd}/nhl_distances_matrix.csv', index=False, encoding='utf-8 sig')
