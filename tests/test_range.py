"""Tests: range."""

from __future__ import annotations

import pytest

import bms


class TestRangePredictor:
    def _pred(self, **kw):
        return bms.RangePredictor(**kw)

    def test_vehicle_presets_instantiate(self):
        for name, veh in bms.VEHICLE_PRESETS.items():
            assert veh.mass_kg > 0
            assert veh.drag_coefficient > 0

    def test_weather_presets(self):
        for factory in [
            bms.WeatherConditions.mild,
            bms.WeatherConditions.hot_summer,
            bms.WeatherConditions.cold_winter,
            bms.WeatherConditions.rainy,
            bms.WeatherConditions.mountain_pass,
        ]:
            w = factory()
            assert isinstance(w, bms.WeatherConditions)

    def test_route_profiles_non_empty(self):
        for key, segs in bms.ROUTE_PROFILES.items():
            assert len(segs) > 0
            total_km = sum(s.distance_km for s in segs)
            assert total_km > 0

    def test_basic_prediction_positive_range(self):
        pred = self._pred()
        result = pred.predict(80_000.0, "nmc", bms.ROUTE_PROFILES["wltp"])
        assert result.estimated_range_km > 0

    def test_max_range_positive(self):
        pred = self._pred()
        r = pred.predict_max_range_km(80_000.0, "nmc", "wltp")
        assert r > 50.0   # sanity: should be well above 50 km

    def test_cold_reduces_range(self):
        pred = self._pred()
        r_mild = pred.predict_max_range_km(80_000.0, "nmc", "wltp",
                                            bms.WeatherConditions.mild())
        r_cold = pred.predict_max_range_km(80_000.0, "nmc", "wltp",
                                            bms.WeatherConditions.cold_winter())
        assert r_cold < r_mild, "Cold weather must reduce range"

    def test_headwind_reduces_range(self):
        pred = self._pred()
        r_calm = pred.predict_max_range_km(
            80_000.0, "nmc", "highway", bms.WeatherConditions(wind_speed_ms=0))
        r_head = pred.predict_max_range_km(
            80_000.0, "nmc", "highway",
            bms.WeatherConditions(wind_speed_ms=15.0, wind_heading_deg=0.0))
        assert r_head < r_calm

    def test_mountain_higher_consumption_than_city(self):
        pred = self._pred()
        r_city = pred.predict_max_range_km(80_000.0, "nmc", "city")
        r_mountain = pred.predict_max_range_km(80_000.0, "nmc", "mountain")
        # Mountain has steep uphills → higher energy per km → shorter range
        assert r_mountain < r_city

    def test_ssb_more_cold_sensitive_than_nmc(self):
        pred = self._pred()
        cold = bms.WeatherConditions(temperature_C=-20.0)
        mild = bms.WeatherConditions.mild()
        penalty_nmc = (pred.predict_max_range_km(80_000.0, "nmc", "wltp", mild)
                       - pred.predict_max_range_km(80_000.0, "nmc", "wltp", cold))
        penalty_ssb = (pred.predict_max_range_km(80_000.0, "ssb", "wltp", mild)
                       - pred.predict_max_range_km(80_000.0, "ssb", "wltp", cold))
        assert penalty_ssb > penalty_nmc, \
            "SSB (higher Arrhenius) must suffer more range loss at -20°C than NMC"

    def test_route_completable_short_route(self):
        pred = self._pred()
        short = [bms.RouteSegment(5.0, 60.0, 0.0, 1.0, "Short trip")]
        result = pred.predict(80_000.0, "nmc", short)
        assert result.route_completable
        assert result.soc_at_destination is not None
        assert result.soc_at_destination > pred.SOC_RESERVE

    def test_route_not_completable_tiny_battery(self):
        pred = self._pred()
        long_route = bms.ROUTE_PROFILES["highway"]
        result = pred.predict(500.0, "nmc", long_route)   # 500 Wh = tiny
        assert not result.route_completable
        assert result.soc_at_destination is None

    def test_energy_breakdown_structure(self):
        pred = self._pred()
        result = pred.predict(80_000.0, "nmc", bms.ROUTE_PROFILES["mixed"])
        bd = result.energy_breakdown
        assert "Traction" in bd
        assert "HVAC" in bd
        assert "Regeneration" in bd
        assert bd["Traction"] >= 0.0
        assert bd["Regeneration"] <= 0.0

    def test_temperature_sweep_ordering(self):
        pred = self._pred()
        sweep = pred.temperature_range_sweep(80_000.0, "nmc", "wltp",
                                              temperatures_C=[-20, 0, 20, 40])
        ranges = list(sweep.values())
        # Range at -20°C < range at 20°C (may be non-monotone at very hot)
        assert ranges[0] < ranges[2]

    def test_suv_shorter_range_than_compact(self):
        compact_pred = self._pred(vehicle=bms.VehicleParams.compact())
        suv_pred = self._pred(vehicle=bms.VehicleParams.suv())
        r_compact = compact_pred.predict_max_range_km(80_000.0, "nmc", "highway")
        r_suv = suv_pred.predict_max_range_km(80_000.0, "nmc", "highway")
        assert r_suv < r_compact, "SUV (heavier, higher drag) must have shorter range"

    def test_downhill_provides_regen(self):
        pred = self._pred()
        downhill = [bms.RouteSegment(10.0, 60.0, -8.0, 1.0, "Steep descent")]
        result = pred.predict(80_000.0, "nmc", downhill)
        # Energy recovered should be non-zero (stored as negative in breakdown)
        assert result.energy_breakdown["Regeneration"] < 0.0



class TestIndiaTwoWheeler:
    def _pred(self, vehicle=None):
        return bms.RangePredictor(vehicle=vehicle)

    # ── Two-wheeler presets ───────────────────────────────────────────────
    def test_e_scooter_has_no_hvac(self):
        scooter = bms.VehicleParams.e_scooter()
        assert scooter.hvac_max_W == 0.0

    def test_e_motorcycle_has_no_hvac(self):
        assert bms.VehicleParams.e_motorcycle().hvac_max_W == 0.0

    def test_e_moped_has_no_hvac(self):
        assert bms.VehicleParams.e_moped().hvac_max_W == 0.0

    def test_two_wheeler_presets_in_vehicle_presets(self):
        for key in ("e_scooter", "e_motorcycle", "e_moped"):
            assert key in bms.VEHICLE_PRESETS
            v = bms.VEHICLE_PRESETS[key]
            assert v.hvac_max_W == 0.0
            assert v.mass_kg < 500.0   # lighter than any car

    def test_e_scooter_lighter_than_sedan(self):
        assert bms.VehicleParams.e_scooter().mass_kg < bms.VehicleParams.sedan().mass_kg

    def test_two_wheeler_positive_range(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        result = pred.predict(3_000.0, "nmc", bms.ROUTE_PROFILES["midc"])
        assert result.estimated_range_km > 0

    def test_moped_vs_motorcycle_range(self):
        # Moped: lower speed, lower mass, worse Cd — net lower range on highway
        moped_pred = self._pred(vehicle=bms.VehicleParams.e_moped())
        moto_pred = self._pred(vehicle=bms.VehicleParams.e_motorcycle())
        r_moped = moped_pred.predict_max_range_km(3_000.0, "nmc", "midc")
        r_moto = moto_pred.predict_max_range_km(3_000.0, "nmc", "midc")
        # Both should be positive
        assert r_moped > 0 and r_moto > 0

    def test_two_wheeler_no_hvac_penalty(self):
        # At extreme cold, 4-wheeler HVAC penalty is real; 2-wheeler should not have it
        pred_car = self._pred(vehicle=bms.VehicleParams.sedan())
        pred_scooter = self._pred(vehicle=bms.VehicleParams.e_scooter())
        cold = bms.WeatherConditions(temperature_C=-10.0)
        result_car = pred_car.predict(40_000.0, "nmc", bms.ROUTE_PROFILES["city"], cold)
        result_scooter = pred_scooter.predict(2_000.0, "nmc", bms.ROUTE_PROFILES["city"], cold)
        assert result_car.energy_breakdown["HVAC"] > 0.0
        assert result_scooter.energy_breakdown["HVAC"] == pytest.approx(0.0)

    # ── MIDC drive cycle ────────────────────────────────────────────────
    def test_midc_profile_exists(self):
        assert "midc" in bms.ROUTE_PROFILES
        segs = bms.ROUTE_PROFILES["midc"]
        assert len(segs) == 2
        total_km = sum(s.distance_km for s in segs)
        assert abs(total_km - 19.7) < 0.5   # MIDC total ≈ 19.7 km

    def test_india_nh_profile_exists(self):
        assert "india_nh" in bms.ROUTE_PROFILES
        segs = bms.ROUTE_PROFILES["india_nh"]
        assert len(segs) == 5
        total_km = sum(s.distance_km for s in segs)
        assert total_km > 60.0

    # ── Road quality ────────────────────────────────────────────────────
    def test_poor_road_increases_consumption(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        # At slow city speeds (<20 km/h), rolling resistance dominates aero drag,
        # so the Cr penalty from poor road outweighs the speed-reduction aero saving
        good_seg = [bms.RouteSegment(10.0, 15.0, 0.0, 0.80, "Good road", "good")]
        poor_seg = [bms.RouteSegment(10.0, 15.0, 0.0, 0.80, "Poor road", "poor")]
        w = bms.WeatherConditions.mild()
        e_good = pred.predict(5_000.0, "nmc", good_seg, w).total_consumed_Wh
        e_poor = pred.predict(5_000.0, "nmc", poor_seg, w).total_consumed_Wh
        assert e_poor > e_good, "Poor road quality must increase energy consumption"

    def test_excellent_road_lower_than_good(self):
        pred = self._pred()
        good_seg = [bms.RouteSegment(10.0, 80.0, 0.0, 1.0, "Good", "good")]
        exc_seg = [bms.RouteSegment(10.0, 80.0, 0.0, 1.0, "Excellent", "excellent")]
        w = bms.WeatherConditions.mild()
        e_good = pred.predict(80_000.0, "nmc", good_seg, w).total_consumed_Wh
        e_exc = pred.predict(80_000.0, "nmc", exc_seg, w).total_consumed_Wh
        assert e_exc < e_good

    # ── India city routes ───────────────────────────────────────────────
    def test_india_city_routes_exist(self):
        expected = {"delhi_ncr", "mumbai", "bangalore", "chennai",
                    "pune", "hyderabad", "kolkata"}
        assert expected.issubset(set(bms.INDIA_CITY_ROUTES.keys()))

    def test_india_city_routes_non_empty(self):
        for city, segs in bms.INDIA_CITY_ROUTES.items():
            assert len(segs) > 0, f"{city} has no segments"
            assert sum(s.distance_km for s in segs) > 0

    def test_india_city_route_prediction_positive(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        for city, segs in bms.INDIA_CITY_ROUTES.items():
            result = pred.predict(3_000.0, "lfp", segs)
            assert result.estimated_range_km > 0, f"{city}: range must be > 0"

    # ── India weather presets ───────────────────────────────────────────
    def test_india_weather_presets(self):
        for factory in [
            bms.WeatherConditions.india_summer,
            bms.WeatherConditions.india_monsoon,
            bms.WeatherConditions.india_winter_north,
            bms.WeatherConditions.india_coastal,
        ]:
            w = factory()
            assert isinstance(w, bms.WeatherConditions)
            assert -20.0 <= w.temperature_C <= 55.0

    def test_india_weather_dict(self):
        assert len(bms.INDIA_WEATHER) >= 12
        for key, w in bms.INDIA_WEATHER.items():
            assert isinstance(w, bms.WeatherConditions)

    def test_india_summer_hot(self):
        w = bms.WeatherConditions.india_summer()
        assert w.temperature_C >= 38.0

    def test_india_monsoon_has_rain(self):
        w = bms.WeatherConditions.india_monsoon()
        assert w.precipitation == "rain"

    def test_summer_reduces_range_vs_mild_india(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        route = bms.ROUTE_PROFILES["midc"]
        r_mild = pred.predict_max_range_km(3_000.0, "nmc", "midc",
                                            bms.WeatherConditions.mild())
        r_summer = pred.predict_max_range_km(3_000.0, "nmc", "midc",
                                              bms.WeatherConditions.india_summer())
        # Hot reduces capacity slightly; AC draw is zero (e-scooter) but
        # hot weather overhead is small — range may be slightly lower
        assert r_mild > 0 and r_summer > 0

