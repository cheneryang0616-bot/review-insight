#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多轮闭环测试：失败之后的行为对不对

这组测试不关心"结论准不准"，只关心：
- 听不懂时会不会硬答（不该）
- 缺参数时会不会崩（不该）
- 样本不足时会不会自动放宽（该）
- 工具异常时会不会照出答案（不该）

改动前基线是 4/9（见 results/eval_loop.json 的历史记录），
这些测试把当时的三类故障固定下来。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.domains import get_domain
from insight.domains.credit import data as credit_data
from insight.domains.review import data as review_data

try:
    credit_data.load()
    HAS_CREDIT = True
except FileNotFoundError:
    HAS_CREDIT = False

# 这条用例要靠"放宽到全量后仍能找到达标套餐"来验证，只有完整评论数据成立
try:
    HAS_FULL_REVIEW = review_data.load().attrs.get("source") == "full"
except Exception:
    HAS_FULL_REVIEW = False


class TestReviewLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(get_domain("review"))

    def test_unparsed_question_asks_instead_of_guessing(self):
        r = self.agent.ask("把 Income 分箱看一下", use_llm=False)
        self.assertEqual(r["status"], "clarification")
        self.assertIn("能做的是", r["answer"])
        self.assertEqual(r["results"], [])

    def test_normal_question_not_misjudged(self):
        r = self.agent.ask("为什么差评这么多？短板在哪？", use_llm=False)
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["grounding"]["passed"])

    def test_entity_question_not_clarified(self):
        # 有实体（店铺）即使意图没命中，也不该反问
        r = self.agent.ask("店铺 4134768 的差评情况怎么样？", use_llm=False)
        self.assertEqual(r["status"], "ok")

    @unittest.skipUnless(HAS_FULL_REVIEW, "需要完整评论数据")
    def test_insufficient_relaxes_and_discloses(self):
        # 该店的套餐每款都不到样本门槛 → 应放宽并显式说明改了口径
        r = self.agent.ask("店铺 4134768 的套餐差评率排名", use_llm=False)
        self.assertEqual(r["status"], "ok")
        self.assertIn("原切片样本不足", r["answer"])
        actions = [t.get("action") for t in r["trajectory"]]
        self.assertTrue(any(a and "放宽切片" in a for a in actions))

    def test_loop_terminates(self):
        r = self.agent.ask("店铺 4134768 的套餐差评率排名", use_llm=False)
        self.assertLessEqual(len(r["trajectory"]), self.agent.max_turns)


@unittest.skipUnless(HAS_CREDIT, "缺少信贷数据")
class TestCreditLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(get_domain("credit"))

    def test_missing_required_param_asks_with_candidates(self):
        r = self.agent.ask("看下这个变量的分箱", use_llm=False)
        self.assertEqual(r["status"], "clarification")
        self.assertIn("feature", r["answer"])
        # 澄清时必须给候选，否则用户不知道能填什么
        self.assertTrue("可选" in r["answer"])

    def test_unknown_field_asks_instead_of_crashing(self):
        r = self.agent.ask("把 Income 分箱看一下", use_llm=False)
        self.assertEqual(r["status"], "clarification")
        self.assertFalse(any(res.get("error") for res in r["results"]))

    def test_tool_health_gate_clean_on_clarification(self):
        r = self.agent.ask("看下这个变量的分箱", use_llm=False)
        self.assertTrue(r["gates"]["passed"])

    def test_valid_question_answers(self):
        r = self.agent.ask("模型效果怎么样？", use_llm=False)
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["grounding"]["passed"])


class TestLoopPlumbing(unittest.TestCase):
    def test_insufficient_flag_set_by_tools(self):
        from insight.domains.review import tools
        df = get_domain("review").load()
        res = tools.call("star_distribution", df, shop="不存在的店")
        self.assertTrue(res.get("insufficient"))
        self.assertEqual(res["facts"].get("n"), 0)

    def test_relax_order(self):
        agent = Agent(get_domain("review"))
        new, dropped = agent._relax({"shop": "x", "year": 2019})
        self.assertEqual(dropped, "year")      # 先放宽年份
        new2, dropped2 = agent._relax({"shop": "x"})
        self.assertEqual(dropped2, "shop")
        _, none = agent._relax({})
        self.assertIsNone(none)


if __name__ == "__main__":
    unittest.main(verbosity=2)
