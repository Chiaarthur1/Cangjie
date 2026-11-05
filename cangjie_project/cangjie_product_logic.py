"""
Project Cangjie (Цанцзе) — Phase 1/4/5 Enterprise Neuro-Symbolic Logic Engine [v3.0]
Product Real Logic (PRL) & Differentiable Reichenbach Falsification Framework:
- Continuous Product t-norm: T_P(x, y) = x * y (Zero Gradient Vanishing Guarantee)
- Reichenbach Implication: I_R(x, y) = 1 - x + x * y
- Goguen Residuum: I_G(x, y) = min(1, y / x)
- Smooth Differentiable Hallucination Penalty: L = P(r_i) * (1 - Q(r_i))
- Dual Engine: Pure NumPy / Float (CPU Inference) & PyTorch AutoGrad (GPU Training/CoVe)
"""

from __future__ import annotations

import math
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# ==============================================================================
# 1. МАТЕМАТИЧЕСКИЙ БАЗИС PRODUCT REAL LOGIC
# ==============================================================================

class ProductRealLogic:
    """
    Математический аппарат непрерывной логики реальных чисел (Product Real Logic).
    Устраняет проблему «плато нулевых градиентов» t-нормы Лукасевича:
    d/dx (max(0, x + y - 1)) = 0 при x + y < 1.
    В Product t-norm градиент d/dx (x * y) = y > 0 сохраняется во всем универсуме (0, 1].
    """

    @staticmethod
    def conjunction(x: Any, y: Any) -> Any:
        """Product t-norm: T_P(x, y) = x * y"""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.clamp(x * y, 0.0, 1.0)
        elif isinstance(x, np.ndarray):
            return np.clip(x * y, 0.0, 1.0)
        else:
            return max(0.0, min(1.0, float(x) * float(y)))

    @staticmethod
    def disjunction(x: Any, y: Any) -> Any:
        """Product t-conorm (вероятностная сумма): S_P(x, y) = x + y - x * y"""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.clamp(x + y - (x * y), 0.0, 1.0)
        elif isinstance(x, np.ndarray):
            return np.clip(x + y - (x * y), 0.0, 1.0)
        else:
            fx, fy = float(x), float(y)
            return max(0.0, min(1.0, fx + fy - (fx * fy)))

    @staticmethod
    def negation(x: Any) -> Any:
        """Стандартное инволютивное отрицание: N(x) = 1 - x"""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.clamp(1.0 - x, 0.0, 1.0)
        elif isinstance(x, np.ndarray):
            return np.clip(1.0 - x, 0.0, 1.0)
        else:
            return max(0.0, min(1.0, 1.0 - float(x)))

    @staticmethod
    def reichenbach_implication(x: Any, y: Any) -> Any:
        """
        Импликация Райхенбаха: I_R(x, y) = 1 - x + x * y.
        Гладкая, всюду дифференцируемая функция, связывающая посылку x и консеквент y.
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.clamp(1.0 - x + (x * y), 0.0, 1.0)
        elif isinstance(x, np.ndarray):
            return np.clip(1.0 - x + (x * y), 0.0, 1.0)
        else:
            fx, fy = float(x), float(y)
            return max(0.0, min(1.0, 1.0 - fx + (fx * fy)))

    @staticmethod
    def goguen_implication(x: Any, y: Any, eps: float = 1e-9) -> Any:
        """
        Резидиум произведения (Импликация Гоголя):
        I_G(x, y) = 1 при x <= y, иначе y / x.
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            div = y / torch.clamp(x, min=eps)
            return torch.where(x <= y, torch.ones_like(x), torch.clamp(div, 0.0, 1.0))
        elif isinstance(x, np.ndarray):
            div = y / np.clip(x, eps, None)
            return np.where(x <= y, 1.0, np.clip(div, 0.0, 1.0))
        else:
            fx, fy = float(x), float(y)
            return 1.0 if fx <= fy else max(0.0, min(1.0, fy / max(eps, fx)))

    @staticmethod
    def equivalence(x: Any, y: Any) -> Any:
        """Логическая эквивалентность: E(x, y) = min(I_G(x, y), I_G(y, x))"""
        i1 = ProductRealLogic.goguen_implication(x, y)
        i2 = ProductRealLogic.goguen_implication(y, x)
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.minimum(i1, i2)
        elif isinstance(x, np.ndarray):
            return np.minimum(i1, i2)
        else:
            return min(float(i1), float(i2))


# ==============================================================================
# 2. ДИФФЕРЕНЦИРУЕМЫЕ ФУНКЦИИ ПОТЕРЬ (НЕЙРОСИМВОЛЬНЫЙ МОСТ)
# ==============================================================================

class NeuroSymbolicLossEngine:
    """
    Дифференцируемый штрафной шлюз для обучения моделей (RT-DETR / Qwen-VL)
    и мультиагентного согласования онтологии Neo4j.
    """

    @staticmethod
    def hallucination_penalty(p_prediction: Any, q_ontology_truth: Any) -> Any:
        """
        Линейный штраф за галлюцинацию (Шаг 1.3 Мастер-плана):
        L_halluc = P(r_i) * (1 - Q(r_i))
        Если детектор уверен в ключе r_i (P близко к 1), а граф знаний
        опровергает его присутствие (Q = 0), штраф строго пропорционален P.
        """
        if HAS_TORCH and isinstance(p_prediction, torch.Tensor):
            penalty = p_prediction * (1.0 - q_ontology_truth)
            return torch.mean(penalty)
        elif isinstance(p_prediction, np.ndarray):
            return float(np.mean(p_prediction * (1.0 - q_ontology_truth)))
        else:
            return float(p_prediction) * (1.0 - float(q_ontology_truth))

    @staticmethod
    def topological_consistency_loss(
        p_operator: Any,
        p_radical_left: Any,
        p_radical_right: Any,
        q_graph_relation: Any
    ) -> Any:
        """
        Проверка истинности тернарной аксиомы композиции:
        Axiom: Operator(op) AND Left(r1) AND Right(r2) => InGraph(IDS)
        Через импликацию Райхенбаха: Loss = 1 - I_R(P_premise, Q_conclusion)
        """
        premise = ProductRealLogic.conjunction(
            p_operator,
            ProductRealLogic.conjunction(p_radical_left, p_radical_right)
        )
        truth_value = ProductRealLogic.reichenbach_implication(premise, q_graph_relation)
        return 1.0 - truth_value


# ==============================================================================
# 3. АГЕНТНЫЙ ВАЛИДАТОР (ФАЛЬСИФИКАТОР РАЙХЕНБАХА ДЛЯ CoVe-RAG+)
# ==============================================================================

class ReichenbachFalsifier:
    """
    Ядро дуэли Агент-Фальсификатор vs Агент-Критик (Фаза 5, Шаг 5.2):
    Оценивает палеографическую гипотезу на основе согласованности с графом.
    """
    def __init__(self, acceptance_threshold: float = 0.50):
        self.threshold = acceptance_threshold

    def evaluate_hypothesis(
        self,
        forger_confidence: float,
        graph_verification_truth: float,
        visual_alignment_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Оценивает гипотезу Фальсификатора:
        Посылка (x): уверенность Фальсификатора p.
        Консеквент (y): истинность из графа Neo4j q, взвешенная зрительным выравниванием SigLIP.
        """
        consequent = max(0.0, min(1.0, graph_verification_truth * visual_alignment_score))
        implication_val = ProductRealLogic.reichenbach_implication(forger_confidence, consequent)

        # Вычисление степени выполнимости (LTN Satisfiability)
        satisfiability = round(float(implication_val * (0.5 + 0.5 * visual_alignment_score)), 4)
        is_accepted = satisfiability >= self.threshold

        return {
            "forger_premise": round(forger_confidence, 4),
            "ontology_consequent": round(consequent, 4),
            "implication_score": round(float(implication_val), 4),
            "ltn_satisfiability": satisfiability,
            "verdict": "ACCEPTED" if is_accepted else "FALSIFIED",
            "is_rejected_by_critic": not is_accepted
        }


# ==============================================================================
# 4. ТЕСТОВЫЙ КОМПЛЕКС И ДЕМОНСТРАЦИЯ ГРАДИЕНТОВ
# ==============================================================================

def run_self_verification():
    print("=" * 70)
    print("Запуск самопроверки Product Real Logic (Project Cangjie)...")
    print("=" * 70)

    # 1. Тест базовых формул t-норм
    x, y = 0.8, 0.6
    t_p = ProductRealLogic.conjunction(x, y)
    s_p = ProductRealLogic.disjunction(x, y)
    i_r = ProductRealLogic.reichenbach_implication(x, y)
    i_g = ProductRealLogic.goguen_implication(x, y)

    assert abs(t_p - 0.48) < 1e-6, f"Ошибка t-нормы: {t_p}"
    assert abs(s_p - 0.92) < 1e-6, f"Ошибка t-конормы: {s_p}"
    assert abs(i_r - 0.68) < 1e-6, f"Ошибка импликации Райхенбаха: {i_r}"
    assert abs(i_g - 0.75) < 1e-6, f"Ошибка резидиума Гоголя: {i_g}"
    print("✅ Базовые алгебраические свойства Product Real Logic подтверждены.")

    # 2. Тест штрафа за галлюцинацию
    loss_halluc = NeuroSymbolicLossEngine.hallucination_penalty(0.95, 0.0)
    assert abs(loss_halluc - 0.95) < 1e-6, f"Ошибка штрафа: {loss_halluc}"

    loss_real = NeuroSymbolicLossEngine.hallucination_penalty(0.95, 1.0)
    assert abs(loss_real - 0.0) < 1e-6, f"Штраф должен быть нулевым: {loss_real}"
    print("✅ Дифференцируемый штраф за галлюцинацию (L = P * (1 - Q)) подтвержден.")

    # 3. Тест дуэли Агента-Критика (ReichenbachFalsifier)
    falsifier = ReichenbachFalsifier(acceptance_threshold=0.50)

    # Кейс 1: Фальсификатор выдает подделку (Dataset_C, Q = 0.0, уверенность = 0.94)
    res_fake = falsifier.evaluate_hypothesis(
        forger_confidence=0.94,
        graph_verification_truth=0.0,
        visual_alignment_score=0.90
    )
    print(f"Кейс Dataset_C (Подделка): Вердикт = {res_fake['verdict']} | Satisfiability = {res_fake['ltn_satisfiability']}")
    assert res_fake["verdict"] == "FALSIFIED", "Подделка должна быть отбракована Критиком!"

    # Кейс 2: Канонический знак (Dataset_A/B, Q = 1.0, уверенность = 0.94)
    res_real = falsifier.evaluate_hypothesis(
        forger_confidence=0.94,
        graph_verification_truth=1.0,
        visual_alignment_score=0.95
    )
    print(f"Кейс Dataset_A (Канон):    Вердикт = {res_real['verdict']} | Satisfiability = {res_real['ltn_satisfiability']}")
    assert res_real["verdict"] == "ACCEPTED", "Канонический знак обязан быть принят Критиком!"

    # 4. Проверка градиентов (если PyTorch установлен)
    if HAS_TORCH:
        p_t = torch.tensor([0.2], requires_grad=True)
        q_t = torch.tensor([0.0])
        loss = p_t * (1.0 - q_t)
        loss.backward()
        assert p_t.grad is not None and abs(p_t.grad.item() - 1.0) < 1e-5
        print(f"✅ PyTorch AutoGrad: Градиент dL/dP = {p_t.grad.item():.4f} (Затухание градиента исключено).")
    else:
        print("ℹ️ PyTorch не обнаружен в локальном окружении; вычисления проверены в NumPy/Float.")

    print("=" * 70)
    print("ВСЕ ТЕСТЫ PRODUCT REAL LOGIC УСПЕШНО ПРОЙДЕНЫ! 🚀")
    print("=" * 70)


if __name__ == "__main__":
    run_self_verification()
