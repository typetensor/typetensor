import { expectTypeOf } from 'expect-type';
import type { Abs, Max, Min, Ceil, Clamp, NormalizeIndex, IntegerPart, IsInteger } from './arithmetic';

expectTypeOf<Abs<5>>().toEqualTypeOf<5>();
expectTypeOf<Abs<-5>>().toEqualTypeOf<5>();
expectTypeOf<Abs<0>>().toEqualTypeOf<0>();
expectTypeOf<Abs<-123>>().toEqualTypeOf<123>();

expectTypeOf<Max<5, 3>>().toEqualTypeOf<5>();
expectTypeOf<Max<3, 5>>().toEqualTypeOf<5>();
expectTypeOf<Max<-5, -3>>().toEqualTypeOf<-3>();
expectTypeOf<Max<0, -1>>().toEqualTypeOf<0>();

expectTypeOf<Min<5, 3>>().toEqualTypeOf<3>();
expectTypeOf<Min<3, 5>>().toEqualTypeOf<3>();
expectTypeOf<Min<-5, -3>>().toEqualTypeOf<-5>();
expectTypeOf<Min<0, -1>>().toEqualTypeOf<-1>();

expectTypeOf<Ceil<10, 3>>().toEqualTypeOf<4>(); // ceil(10/3) = 4
expectTypeOf<Ceil<9, 3>>().toEqualTypeOf<3>(); // ceil(9/3) = 3
expectTypeOf<Ceil<11, 3>>().toEqualTypeOf<4>(); // ceil(11/3) = 4
expectTypeOf<Ceil<1, 1>>().toEqualTypeOf<1>(); // ceil(1/1) = 1

expectTypeOf<Clamp<5, 0, 10>>().toEqualTypeOf<5>();
expectTypeOf<Clamp<-5, 0, 10>>().toEqualTypeOf<0>();
expectTypeOf<Clamp<15, 0, 10>>().toEqualTypeOf<10>();
expectTypeOf<Clamp<3, 3, 3>>().toEqualTypeOf<3>();

expectTypeOf<NormalizeIndex<5, 10>>().toEqualTypeOf<5>();
expectTypeOf<NormalizeIndex<-1, 10>>().toEqualTypeOf<9>(); // 10 + (-1) = 9
expectTypeOf<NormalizeIndex<-5, 10>>().toEqualTypeOf<5>(); // 10 + (-5) = 5
expectTypeOf<NormalizeIndex<0, 10>>().toEqualTypeOf<0>();

// IntegerPart tests
expectTypeOf<IntegerPart<4>>().toEqualTypeOf<4>();
expectTypeOf<IntegerPart<4.5>>().toEqualTypeOf<4>();
expectTypeOf<IntegerPart<4.9>>().toEqualTypeOf<4>();
expectTypeOf<IntegerPart<0.5>>().toEqualTypeOf<0>();
expectTypeOf<IntegerPart<-3.7>>().toEqualTypeOf<-3>();
expectTypeOf<IntegerPart<123>>().toEqualTypeOf<123>();

// IsInteger tests
expectTypeOf<IsInteger<4>>().toEqualTypeOf<true>();
expectTypeOf<IsInteger<0>>().toEqualTypeOf<true>();
expectTypeOf<IsInteger<-5>>().toEqualTypeOf<true>();
expectTypeOf<IsInteger<4.5>>().toEqualTypeOf<false>();
expectTypeOf<IsInteger<0.1>>().toEqualTypeOf<false>();
expectTypeOf<IsInteger<2.4>>().toEqualTypeOf<false>();
expectTypeOf<IsInteger<-3.7>>().toEqualTypeOf<false>();
