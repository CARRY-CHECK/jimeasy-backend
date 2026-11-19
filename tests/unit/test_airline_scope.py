import unittest

from app.query_gold_similarity import matches_where
from app.retrieval import airline_orgs, airline_where, retrieval_where


def conditions(dep_place=None, dep_airline=None, arr_place=None, arr_airline=None):
    return {'departure': {'place': dep_place, 'airline': dep_airline},
            'arrival': {'place': arr_place, 'airline': arr_airline}}


class AirlineScopeTests(unittest.TestCase):
    def test_default_where_is_unchanged_and_airline_scope_is_opt_in(self):
        cond = conditions('인천', '대한항공', '도쿄')
        base = retrieval_where(None, cond)
        self.assertEqual(base, {'country': {'$in': ['KR', 'JP']}})
        self.assertEqual(retrieval_where(None, cond, airline_scope=False), base)
        scoped = retrieval_where(None, cond, airline_scope=True)
        self.assertEqual(scoped['$and'][0], base)
        self.assertEqual(scoped['$and'][1], airline_where(cond))

    def test_no_airline_or_unmapped_airline_keeps_every_document(self):
        for cond in (conditions('인천', None, '도쿄'), conditions('인천', '에어프랑스', '파리')):
            self.assertEqual(airline_orgs(cond), [])
            self.assertIsNone(airline_where(cond))
            self.assertEqual(retrieval_where(None, cond, airline_scope=True), retrieval_where(None, cond))

    def test_return_leg_airline_is_included_and_duplicates_removed(self):
        self.assertEqual(airline_orgs(conditions('인천', '진에어', '도쿄', '아시아나항공')), ['진에어', '아시아나항공'])
        self.assertEqual(airline_orgs(conditions('인천', '대한항공', '도쿄', '대한항공')), ['대한항공'])
        self.assertEqual(airline_orgs(conditions('인천', ' ana ')), ['ANA'])

    def test_where_keeps_non_airline_sources_and_only_the_requested_airline(self):
        where = airline_where(conditions('인천', '대한항공'))
        government = {'source_type': 'admin_rule', 'org_name': '국토교통부'}
        airport = {'source_type': 'airport', 'org_name': '인천국제공항공사'}
        mine = {'source_type': 'airline', 'org_name': '대한항공'}
        other = {'source_type': 'airline', 'org_name': '진에어'}
        self.assertTrue(all(matches_where(m, where) for m in (government, airport, mine)))
        self.assertFalse(matches_where(other, where))

    def test_two_airlines_use_in_operator(self):
        where = airline_where(conditions('인천', '진에어', '도쿄', '아시아나항공'))
        self.assertEqual(where['$or'][1], {'org_name': {'$in': ['진에어', '아시아나항공']}})
        self.assertFalse(matches_where({'source_type': 'airline', 'org_name': '대한항공'}, where))


if __name__ == '__main__':
    unittest.main()
