import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob



'''
 Ejemplo de codigo para creacion de tablas para utilir posteriormente segun necesidades 
'''
# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Cargar todos los archivos CSV
data_path = Path("data")
all_files = glob.glob(str(data_path / "season-*.csv"))

# Leer y combinar todos los datasets
df_list = []
for file in all_files:
    df_temp = pd.read_csv(file)
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)

# Convertir la columna Date a datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')

# Extraer mes y nombre del mes
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')

# Crear columna de periodo de temporada
def classify_season_period(month):
    if month in [8, 9]:
        return 'Inicio (Ago-Sep)'
    elif month in [12, 1]:
        return 'Mitad (Dic-Ene)'
    elif month in [4, 5]:
        return 'Final (Abr-May)'
    elif month in [10, 11]:
        return 'Otoño (Oct-Nov)'
    elif month in [2, 3]:
        return 'Invierno (Feb-Mar)'
    else:
        return 'Otro'

df['Season_Period'] = df['Month'].apply(classify_season_period)

# ===== ANÁLISIS 1: Goles promedio por mes =====
print("=" * 60)
print("ANÁLISIS DE TENDENCIAS ESTACIONALES")
print("=" * 60)

# Calcular goles totales por partido
df['Total_Goals'] = df['FTHG'] + df['FTAG']

# Agrupar por mes
monthly_stats = df.groupby('Month').agg({
    'Total_Goals': 'mean',
    'FTHG': 'mean',
    'FTAG': 'mean',
    'FTR': lambda x: (x == 'H').sum() / len(x) * 100  # % victorias locales
}).round(2)

monthly_stats.columns = ['Goles_Promedio_Total', 'Goles_Local', 'Goles_Visitante', 'Pct_Victorias_Local']

print("\n📊 ESTADÍSTICAS POR MES:")
print(monthly_stats)

# ===== ANÁLISIS 2: Por periodo de temporada =====
period_stats = df.groupby('Season_Period').agg({
    'Total_Goals': 'mean',
    'FTHG': 'mean',
    'FTAG': 'mean',
    'FTR': [
        lambda x: (x == 'H').sum() / len(x) * 100,  # % victorias locales
        lambda x: (x == 'D').sum() / len(x) * 100,  # % empates
        lambda x: (x == 'A').sum() / len(x) * 100   # % victorias visitantes
    ]
}).round(2)

period_stats.columns = ['Goles_Promedio', 'Goles_Local', 'Goles_Visitante', 'Pct_Victorias_Local', 'Pct_Empates', 'Pct_Victorias_Visitante']

print("\n📅 ESTADÍSTICAS POR PERIODO DE TEMPORADA:")
print(period_stats)

# ===== VISUALIZACIONES =====

# Gráfico 1: Goles promedio por mes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Goles totales por mes
ax1 = axes[0, 0]
monthly_stats['Goles_Promedio_Total'].plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Goles Promedio por Mes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Mes')
ax1.set_ylabel('Goles Promedio por Partido')
ax1.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Goles local vs visitante por mes
ax2 = axes[0, 1]
monthly_stats[['Goles_Local', 'Goles_Visitante']].plot(kind='bar', ax=ax2)
ax2.set_title('Goles Local vs Visitante por Mes', fontsize=14, fontweight='bold')
ax2.set_xlabel('Mes')
ax2.set_ylabel('Goles Promedio')
ax2.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
ax2.legend(['Local', 'Visitante'])
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: % Victorias locales por mes
ax3 = axes[1, 0]
monthly_stats['Pct_Victorias_Local'].plot(kind='line', marker='o', ax=ax3, color='green', linewidth=2)
ax3.set_title('% Victorias Locales por Mes', fontsize=14, fontweight='bold')
ax3.set_xlabel('Mes')
ax3.set_ylabel('% Victorias Locales')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
ax3.grid(True, alpha=0.3)
ax3.axhline(y=monthly_stats['Pct_Victorias_Local'].mean(), color='red', linestyle='--', label='Promedio')
ax3.legend()

# Subplot 4: Comparación por periodo de temporada
ax4 = axes[1, 1]
period_stats[['Pct_Victorias_Local', 'Pct_Empates', 'Pct_Victorias_Visitante']].plot(kind='bar', ax=ax4, stacked=True)
ax4.set_title('Distribución de Resultados por Periodo', fontsize=14, fontweight='bold')
ax4.set_xlabel('Periodo de Temporada')
ax4.set_ylabel('% de Partidos')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.legend(['Victoria Local', 'Empate', 'Victoria Visitante'])
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('tendencias_estacionales.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico guardado como 'tendencias_estacionales.png'")
plt.show()

# ===== ANÁLISIS ESTADÍSTICO =====
print("\n🔍 HALLAZGOS CLAVE:")
print("-" * 60)

# Mes con más goles
mes_max_goles = monthly_stats['Goles_Promedio_Total'].idxmax()
meses_nombres = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
print(f"🔥 Mes con más goles: {meses_nombres[mes_max_goles]} ({monthly_stats.loc[mes_max_goles, 'Goles_Promedio_Total']:.2f} goles/partido)")

# Mes con menos goles
mes_min_goles = monthly_stats['Goles_Promedio_Total'].idxmin()
print(f"❄️  Mes con menos goles: {meses_nombres[mes_min_goles]} ({monthly_stats.loc[mes_min_goles, 'Goles_Promedio_Total']:.2f} goles/partido)")

# Mayor ventaja local
mes_max_ventaja = monthly_stats['Pct_Victorias_Local'].idxmax()
print(f"🏠 Mayor ventaja local: {meses_nombres[mes_max_ventaja]} ({monthly_stats.loc[mes_max_ventaja, 'Pct_Victorias_Local']:.1f}% victorias locales)")

# Comparación periodos
periodo_max_goles = period_stats['Goles_Promedio'].idxmax()
print(f"⚽ Periodo más ofensivo: {periodo_max_goles} ({period_stats.loc[periodo_max_goles, 'Goles_Promedio']:.2f} goles/partido)")

print("\n" + "=" * 60)