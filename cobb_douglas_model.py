"""
Модель экономической системы на основе производственной функции Кобба-Дугласа

Производственная функция Кобба-Дугласа имеет вид:
Y = A * L^α * K^β

где:
- Y - выпуск (output)
- A - общая факторная производительность (total factor productivity)
- L - труд (labor)
- K - капитал (capital)
- α - эластичность выпуска по труду
- β - эластичность выпуска по капиталу
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings


@dataclass
class CobbDouglasParameters:
    """Параметры производственной функции Кобба-Дугласа"""
    A: float = 1.0  # Общая факторная производительность
    alpha: float = 0.3  # Эластичность по труду
    beta: float = 0.7  # Эластичность по капиталу
    
    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Параметры alpha и beta должны быть положительными")
        if self.A <= 0:
            raise ValueError("Параметр A должен быть положительным")


class CobbDouglasEconomy:
    """
    Класс модели экономической системы с производственной функцией Кобба-Дугласа
    """
    
    def __init__(self, params: CobbDouglasParameters = None):
        """
        Инициализация модели
        
        Args:
            params: Параметры производственной функции
        """
        self.params = params or CobbDouglasParameters()
        self.history = {
            'time': [],
            'L': [],
            'K': [],
            'Y': [],
            'A': [],
            'w': [],  # заработная плата
            'r': []   # процентная ставка
        }
    
    def production_function(self, L: float, K: float, A: Optional[float] = None) -> float:
        """
        Вычисляет выпуск по производственной функции Кобба-Дугласа
        
        Args:
            L: Количество труда
            K: Количество капитала
            A: Общая факторная производительность (по умолчанию из параметров)
            
        Returns:
            Y: Выпуск
        """
        A = A if A is not None else self.params.A
        return A * (L ** self.params.alpha) * (K ** self.params.beta)
    
    def marginal_product_labor(self, L: float, K: float, A: Optional[float] = None) -> float:
        """
        Предельный продукт труда (MPL)
        MPL = ∂Y/∂L = α * A * L^(α-1) * K^β = α * Y / L
        """
        A = A if A is not None else self.params.A
        Y = self.production_function(L, K, A)
        return self.params.alpha * Y / L if L > 0 else 0
    
    def marginal_product_capital(self, L: float, K: float, A: Optional[float] = None) -> float:
        """
        Предельный продукт капитала (MPK)
        MPK = ∂Y/∂K = β * A * L^α * K^(β-1) = β * Y / K
        """
        A = A if A is not None else self.params.A
        Y = self.production_function(L, K, A)
        return self.params.beta * Y / K if K > 0 else 0
    
    def returns_to_scale(self) -> str:
        """
        Определяет тип отдачи от масштаба
        """
        sum_elasticity = self.params.alpha + self.params.beta
        
        if np.isclose(sum_elasticity, 1.0):
            return "constant"  # постоянная отдача от масштаба
        elif sum_elasticity > 1.0:
            return "increasing"  # возрастающая отдача от масштаба
        else:
            return "decreasing"  # убывающая отдача от масштаба
    
    def optimal_factor_demand(self, w: float, r: float, target_output: float) -> Tuple[float, float]:
        """
        Вычисляет оптимальный спрос на факторы производства при минимизации издержек
        
        Args:
            w: Заработная плата
            r: Процентная ставка (цена капитала)
            target_output: Целевой объем выпуска
            
        Returns:
            (L*, K*): Оптимальные количества труда и капитала
        """
        A = self.params.A
        alpha = self.params.alpha
        beta = self.params.beta
        
        # Формулы для оптимального спроса на факторы
        # Из условия минимизации издержек: MPL/w = MPK/r
        # и производственного ограничения: Y = A * L^α * K^β
        
        factor = (target_output / A) ** (1 / (alpha + beta))
        ratio = (alpha * r) / (beta * w)
        
        L_opt = factor * (ratio ** (beta / (alpha + beta)))
        K_opt = factor * (ratio ** (-alpha / (alpha + beta)))
        
        return L_opt, K_opt
    
    def simulate_growth(self, 
                       initial_L: float, 
                       initial_K: float,
                       n: float,  # темп роста населения
                       s: float,  # норма сбережений
                       delta: float,  # норма выбытия капитала
                       g_A: float,  # темп роста производительности
                       periods: int) -> dict:
        """
        Симулирует экономический рост во времени (модель Солоу с Кобба-Дугласом)
        
        Args:
            initial_L: Начальное количество труда
            initial_K: Начальный запас капитала
            n: Темп роста населения
            s: Норма сбережений
            delta: Норма выбытия капитала
            g_A: Темп роста общей факторной производительности
            periods: Количество периодов для симуляции
            
        Returns:
            Dictionary с историей переменных
        """
        L = initial_L
        K = initial_K
        A = self.params.A
        
        # Очистка истории
        self.history = {key: [] for key in self.history.keys()}
        
        for t in range(periods):
            Y = self.production_function(L, K, A)
            w = self.marginal_product_labor(L, K, A)
            r = self.marginal_product_capital(L, K, A)
            
            # Сохранение в историю
            self.history['time'].append(t)
            self.history['L'].append(L)
            self.history['K'].append(K)
            self.history['Y'].append(Y)
            self.history['A'].append(A)
            self.history['w'].append(w)
            self.history['r'].append(r)
            
            # Динамика факторов
            I = s * Y  # Инвестиции
            K = (1 - delta) * K + I  # Обновление капитала
            L = L * (1 + n)  # Рост труда
            A = A * (1 + g_A)  # Рост производительности
        
        return self.history
    
    def plot_results(self, variables: List[str] = None):
        """
        Визуализация результатов симуляции
        
        Args:
            variables: Список переменных для отображения
        """
        if not self.history['time']:
            warnings.warn("Нет данных для отображения. Запустите simulate_growth сначала.")
            return
        
        if variables is None:
            variables = ['Y', 'K', 'L']
        
        fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)))
        if len(variables) == 1:
            axes = [axes]
        
        for ax, var in zip(axes, variables):
            if var in self.history:
                ax.plot(self.history['time'], self.history[var], label=var, linewidth=2)
                ax.set_xlabel('Время (t)', fontsize=12)
                ax.set_ylabel(var, fontsize=12)
                ax.set_title(f'Динамика {var}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_production_surface(self, L_range: Tuple[float, float], K_range: Tuple[float, float], 
                               resolution: int = 50):
        """
        Построение 3D поверхности производственной функции
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        L_vals = np.linspace(L_range[0], L_range[1], resolution)
        K_vals = np.linspace(K_range[0], K_range[1], resolution)
        L_grid, K_grid = np.meshgrid(L_vals, K_vals)
        
        Y_grid = self.production_function(L_grid, K_grid)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(L_grid, K_grid, Y_grid, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Труд (L)', fontsize=12)
        ax.set_ylabel('Капитал (K)', fontsize=12)
        ax.set_zlabel('Выпуск (Y)', fontsize=12)
        ax.set_title('Производственная функция Кобба-Дугласа', fontsize=14)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()


def main():
    """Пример использования модели"""
    
    print("=" * 60)
    print("МОДЕЛЬ ЭКОНОМИЧЕСКОЙ СИСТЕМЫ КОББА-ДУГЛАСА")
    print("=" * 60)
    
    # Создание модели с параметрами по умолчанию
    params = CobbDouglasParameters(A=1.0, alpha=0.3, beta=0.7)
    economy = CobbDouglasEconomy(params)
    
    print(f"\nПараметры модели:")
    print(f"  A (производительность) = {params.A}")
    print(f"  α (эластичность по труду) = {params.alpha}")
    print(f"  β (эластичность по капиталу) = {params.beta}")
    print(f"  Отдача от масштаба: {economy.returns_to_scale()}")
    
    # Пример расчета выпуска
    L_example = 100
    K_example = 100
    Y_example = economy.production_function(L_example, K_example)
    
    print(f"\nПример расчета:")
    print(f"  При L = {L_example}, K = {K_example}")
    print(f"  Выпуск Y = {Y_example:.2f}")
    
    # Предельные продукты
    mpl = economy.marginal_product_labor(L_example, K_example)
    mpk = economy.marginal_product_capital(L_example, K_example)
    
    print(f"\nПредельные продукты:")
    print(f"  MPL (предельный продукт труда) = {mpl:.4f}")
    print(f"  MPK (предельный продукт капитала) = {mpk:.4f}")
    
    # Оптимальный спрос на факторы
    w = 1.0
    r = 0.05
    target_Y = 150
    L_opt, K_opt = economy.optimal_factor_demand(w, r, target_Y)
    
    print(f"\nОптимальный спрос на факторы (при w={w}, r={r}, Y={target_Y}):")
    print(f"  L* = {L_opt:.2f}")
    print(f"  K* = {K_opt:.2f}")
    
    # Симуляция экономического роста
    print(f"\nЗапуск симуляции экономического роста...")
    history = economy.simulate_growth(
        initial_L=100,
        initial_K=100,
        n=0.02,      # 2% рост населения
        s=0.2,       # 20% норма сбережений
        delta=0.05,  # 5% выбытие капитала
        g_A=0.01,    # 1% рост производительности
        periods=50
    )
    
    print(f"\nРезультаты симуляции (первые 5 и последние 5 периодов):")
    print(f"{'Период':<8} {'L':<10} {'K':<10} {'Y':<10} {'w':<10} {'r':<10}")
    print("-" * 58)
    
    for i in list(range(5)) + list(range(45, 50)):
        print(f"{i:<8} {history['L'][i]:<10.2f} {history['K'][i]:<10.2f} "
              f"{history['Y'][i]:<10.2f} {history['w'][i]:<10.4f} {history['r'][i]:<10.4f}")
    
    # Построение графиков
    print(f"\nПостроение графиков...")
    try:
        economy.plot_results(['Y', 'K', 'L'])
        print("Графики динамики построены!")
    except Exception as e:
        print(f"Не удалось построить графики: {e}")
    
    # Построение 3D поверхности
    print(f"\nПостроение 3D поверхности производственной функции...")
    try:
        economy.plot_production_surface(
            L_range=(10, 200),
            K_range=(10, 200)
        )
        print("3D поверхность построена!")
    except Exception as e:
        print(f"Не удалось построить 3D поверхность: {e}")
    
    print("\n" + "=" * 60)
    print("Модель готова к использованию!")
    print("=" * 60)


if __name__ == "__main__":
    main()
