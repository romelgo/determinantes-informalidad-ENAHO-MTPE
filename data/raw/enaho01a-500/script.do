
*************************************
***** ESTADISTICAS DE EMPLEO 2015 - 2025 ********
*************************************

* Utilizamos el archivo del capítulo 500 de la ENAHO.

use Enaho01a-2015-500 al Enaho01a-2025-500, clear

/* Eliminamos los casos que no cuentan con información para el empleo
 para ello utilizamos la variable p500i que coresponde al código del
 informante del capítulo 500. Cuando su valor es "00" es porque no se
 pudo obtener información para la persona */
 
 keep if p500i != "00"
 
 /* Generamos variables para el cálculo de indicadores por
    ámbitos geográficos
	
	Area de residencia (urbano / rural)
	Región natural (costa / sierra / selva)
	Departamento (24 departamentos + Prov. Callao + Lima Metropolitana) */
 
 
* Area de residencia
recode estrato (1/5=1) (6/8=2), g(area)
label define area 1 "Urbano" 2 "Rural"
label value area area
label var area "Area de residencia"
 
* Region natural
recode dominio (1/3=1) (8=1) (4/6=2) (7=3), g(region)
label define region 1 "Costa" 2 "Sierra" 3 "Selva"
label value region region
label var region "Region natural"
 
* Departamento
g dep=real(substr(ubigeo,1,2))
replace dep=dep+1 if dep>=16
replace dep=16 if substr(ubigeo,1,4) !="1501" & dep==15
label define dep ///
1 "Amazonas" /// 
2 "Ancash" ///
3 "Apurímac" ///
4 "Arequipa" ///
5 "Ayacucho" ///
6 "Cajamarca" ///
7 "Prov. Const. del Callao" ///
8 "Cusco" ///
9 "Huancavelica" ///
10 "Huánuco" ///
11 "Ica" ///
12 "Junín" ///
13 "La Libertad" ///
14 "Lambayeque" ///
15 "Lima Metropolitana" ///
16 "Lima" ///
17 "Loreto" ///
18 "Madre de Dios" ///
19 "Moquegua" ///
20 "Pasco" ///
21 "Piura" ///
22 "Puno" ///
23 "San Martín" ///
24 "Tacna" ///
25 "Tumbes" ///
26 "Ucayali", replace

label value dep dep
label var dep "departamento"

* Grupo de edad
recode p208a (14/24=1) (25/44=2) (45/max=3), g(gedad)
label define gedad 1 "De 14 a 24 años" 2 "De 25 a 44 años" 3 "De 45 a mas años"
label value gedad gedad
label var gedad "Grupos de edad"


/* El cálculo de estadísticas de empleo se realiza con base en los
   residente habitaules del hogar */ 

g residente = (p204==1 & p205==2) | (p204==2 & p206==1)

* grabamos capitulo 500 con las nuevas variables
save, replace

/* ESTIMACIÓN DE LA PEA
  
   La PEA está conformada por los ocupados y desempleados residentes
   en el hogar.
   Para identificar a esta población utilizamos la variables
   ocu500 que define 4 categorías:
   
   1 ocupado                |
   2 desocupado abierto     |---> PEA   
   3 desocupado oculto   
   4 no pea
*/

* Por area de residencia
tab area [iweight = fac500a] if ocu500<3 & residente==1

* Por region natural
tab region [iweight = fac500a] if ocu500<3 & residente==1

* Por departamento
tab dep [iweight = fac500a] if ocu500<3 & residente==1

** ESTIMACIÓN DE LA TASA DE ACTIVIDAD

/* La Tasa de actividad es el cociente entre la PEA y la PET (Población en edad de trabajar */

drop tasa_actividad
g tasa_actividad = ocu500<3
lab var tasa_actividad "Tasa de Actividad"

* Por area de residencia
tabstat tasa_actividad [aw=fac500a] if residente==1, s(mean) by(area)

* Por region natural
tabstat tasa_actividad [aw=fac500a] if residente==1, s(mean) by(region)

* Por departamento
tabstat tasa_actividad [aw=fac500a] if residente==1, s(mean) by(dep)

** ESTIMACIÓN DE LA POBLACIÓN ECONOMICAMENTE ACTIVA OCUPADA

* Por area de residencia
tab area [iweight = fac500a] if ocu500==1 & residente==1

* Por region natural
tab region [iweight = fac500a] if ocu500==1 & residente==1

* Por departamento
tab dep [iweight = fac500a] if ocu500==1 & residente==1


** ESTIMACIÓN DE LA TASA DE DESEMPLEO ABIERTO EN EL AREA URBANA

/* La Tasa de desempleo abierto es el cociente entre desocupado abierto
   y (ocupado + desocupado abierto) 
   
              desocupado abierto
   TDA = ----------------------------
         ocupado + desocupado abierto
   */


g tda = ocu500==2
label var tda "Tasa de desempleo abierto"   

* Por sexo
tabstat tda [aweight=fac500a] if residente==1 & (ocu500==1 | ocu500==2) & area==1, s(mean) by(p207)

* Por grupo de edad
tabstat tda [aweight=fac500a] if residente==1 & (ocu500==1 | ocu500==2) & area==1, s(mean) by(gedad)


* ESTIMACIÓN DE LA DISTRIBUCIÓN DE LA POBLACIÓN ECONOMICAMENTE NO ACTIVA

* Otro Grupo de edad
recode p208a (14/24=1) (25/44=2) (45/64=3) (65/max=4), g(ogedad)
label define ogedad 1 "De 14 a 24 años" 2 "De 25 a 44 años" 3 "De 45 a 64 años" 4 "De 65 a mas años"
label value ogedad ogedad
label var ogedad "Grupos de edad"

* variable no_pea
g no_pea = ocu500==3 | ocu500==4

* Por sexo
tab p207 no_pea [iw=fac500a] if residente==1, col nofreq

* Por grupo de edad
tab ogedad no_pea [iw=fac500a] if residente==1, col nofreq

** ESTIMACIÓN DE LA TASA DE INACTIVIDAD

/* La Tasa de inactividad es el cociente entre la NO PEA y la PET
   
   Siendo PET = PEA + NO PEA, la Tasa de Inactividad es el complemento
   de la Tasa de Actividad */

g tasa_inactividad = ocu500==3 | ocu500==4
lab var tasa_inactividad "Tasa de Inactividad"

* Por area de residencia
tabstat tasa_inactividad [aw=fac500a] if residen
te==1, s(mean) by(area)

* Por region natural
tabstat tasa_inactividad [aw=fac500a] if residente==1, s(mean) by(region)

* Por departamento
tabstat tasa_inactividad [aw=fac500a] if residente==1, s(mean) by(dep)


** ESTIMACIÓN DE LA TASA DE INFORMALIDAD

/* Para la estimación de la tasa de informalidad, utilizamos la variable
   ocupinf donde 1 = empleo informal y 2 = empleo formal */
   
g tei=ocupinf==1 if ocupinf != .
label var tei "Tasa de empleo informal"

* Por area de residencia
tabstat tei [aw=fac500a] if residente==1, s(mean) by(area)

* Por region natural
tabstat tei [aw=fac500a] if residente==1, s(mean) by(region)

* Por departamento
tabstat tei [aw=fac500a] if residente==1, s(mean) by(dep)
   
   
   