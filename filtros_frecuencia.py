#formato 
#    sample rate, bit depth, channels
#Archivo wav
#Leer wav en scipy
#Escribir wav en scipy
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

frecuencia_muestreo, muestras = wavfile.read("violin.wav")

print(frecuencia_muestreo)

print("Tipo: " + str(type(muestras)))
print("Dtype (bithdept): " + str(muestras.dtype))
print("shape = " + str(muestras.shape))
canales = 1
if len(muestras.shape) == 1:
    print("# Canales = 1")
else:
    print("# Canales = " + str(muestras.shape[1]))
    canales = muestras.shape[1]
duracion = muestras.shape[0] / frecuencia_muestreo
print("duracion:  " + "{:.2f}".format(duracion) + " segs")

tiempos = np.linspace(0., duracion, muestras.shape[0])

figura, ejes = plt.subplots(2,4)
if canales == 1:
    ejes[0,0].plot(tiempos, muestras, label="Canal mono")
else:
    ejes[0,0].plot(tiempos, muestras[:, 0], label="Izquierdo")
    ejes[0,0].plot(tiempos, muestras[:, 1], label="Derecho")

ejes[0,0].legend()
ejes[0,0].set(xlabel = "Tiempo (S)", ylabel = "Amplitud")

if canales > 1:
    data = muestras[:,0]
else:
    data = muestras

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

ejes[1,0].plot(frecuencias, np.abs(transformada), label = "espectro original")
ejes[1,0].legend()
ejes[1,0].set(xlabel = "Frecuencia (Hz)", ylabel = "Amplitud")

#Filtro pasa bajas
pasa_bajas = transformada.copy()
pasa_bajas[frecuencias > 2500] *= 0

ejes[1,1].plot(frecuencias, np.abs(pasa_bajas), label = "Espectro filtrado, pasa bajas")
ejes[1,1].legend()
ejes[1,1].set(xlabel="Frecuencia (Hz)", ylabel = "Amplitud")

pasa_bajas_data = np.fft.irfft(pasa_bajas)

ejes[0,1].plot(tiempos,pasa_bajas_data, label="Audio con pasabajas")
ejes[0,1].legend()
ejes[0,1].set(xlabel="Tiempo (S)", ylabel="Amplitud")

write("pasa_bajas.wav", frecuencia_muestreo, pasa_bajas_data.astype(np.int16))

#Detiene bandas (band stop)
detiene_bandas = transformada.copy()
#aplicar 2 filtros
#detiene_bandas = pasa_bajas.copy()
indices = ((frecuencias > 500) & (frecuencias < 1000))
detiene_bandas[indices] *= 0
detiene_bandas_data = np.fft.irfft(detiene_bandas)

ejes[0,2].plot(tiempos,detiene_bandas_data, label="Audio con detiene bandas")
ejes[0,2].legend()
ejes[0,2].set(xlabel="Tiempo (S)", ylabel="Amplitud")

ejes[1,2].plot(frecuencias,np.abs(detiene_bandas), label="Espectro detiene bandas")
ejes[1,2].legend()
ejes[1,2].set(xlabel="Frecuencia (Hz)", ylabel="Amplitud")

write("detiene_bandas.wav", frecuencia_muestreo, detiene_bandas_data.astype(np.int16))

#Filtro pasa altas
pasa_altas = transformada.copy()
pasa_altas[frecuencias < 400] *= 0

ejes[1,3].plot(frecuencias, np.abs(pasa_altas), label = "Espectro filtrado, pasa altas")
ejes[1,3].legend()
ejes[1,3].set(xlabel="Frecuencia (Hz)", ylabel = "Amplitud")

pasa_altas_data = np.fft.irfft(pasa_altas)

ejes[0,3].plot(tiempos,pasa_altas_data, label="Audio con pasa altas")
ejes[0,3].legend()
ejes[0,3].set(xlabel="Tiempo (S)", ylabel="Amplitud")

write("pasa_altas.wav", frecuencia_muestreo, pasa_altas_data.astype(np.int16))

plt.show()
