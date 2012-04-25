#include "gtest/gtest.h"
#include "dlib/matrix.h"
#include "collaborativeFiltering.h";
#include <tr1/memory>
#include "dlib/optimization.h"

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

using dlib::matrix;
using dlib::randm;
using std::cout;
using std::endl;


//TEST(Matrix, Fold) {
//	matrix<double, 3, 2> m1;
//	m1 = 1,2,
//		3,4,
//		5,6;
//	matrix<double, 2, 3> m2;
//	m2 = 7,8,9,
//		10,11,12;
//	matrix<double, 12, 1> folded;
//	folded = com_firfi::GDFunc<matrix<double> >::fold(m1,m2);
//	matrix<double, 12, 1> fict_folded;
//	fict_folded = 1,2,3,4,5,6,7,8,9,10,11,12;
//	ASSERT_TRUE(folded == fict_folded);
//}

//TEST(Matrix, Unfold) {
//	matrix<double, 12, 1> fict_folded;
//	fict_folded = 1,2,3,4,5,6,7,8,9,10,11,12;
//	matrix<double, 3, 2> m1 = com_firfi::GDFunc<matrix<double> >::unfold(fict_folded,3,2,0);
//	matrix<double, 2, 3> m2 = com_firfi::GDFunc<matrix<double> >::unfold(fict_folded,2,3,m1.nc()*m1.nr());
//	matrix<double, 3, 2> fict_m1;
//	fict_m1 = 1,2,
//			3,4,
//			5,6;
//	matrix<double, 2, 3> fict_m2;
//	fict_m2 = 7,8,9,
//			10,11,12;
//	ASSERT_TRUE(m1 == fict_m1);
//	ASSERT_TRUE(m2 == fict_m2);
//}

TEST(Matrix, Munfold) {
	matrix<double, 12, 1> fict_folded;
	fict_folded = 1,2,3,4,5,6,7,8,9,10,11,12;
	matrix<double, 3, 2> m1 = dlib::munfold(fict_folded,0,3,2);
	matrix<double, 2, 3> m2 = dlib::munfold(fict_folded,6,2,3);
	matrix<double, 3, 2> fict_m1;
	fict_m1 = 1,2,
			3,4,
			5,6;
	matrix<double, 2, 3> fict_m2;
	fict_m2 = 7,8,9,
			10,11,12;
	ASSERT_TRUE(m1 == fict_m1);
	ASSERT_TRUE(m2 == fict_m2);
}


typedef std::vector<std::multimap<double, int, std::greater<double> > > recommends;
TEST(Util, find_n_max) {
	typedef std::pair<double, int> pair_t;
	typedef std::multimap<double, int, std::greater<double> >  map_t;
	static const long num_users = 4;
	static const long num_movies = 5;
	matrix<double, num_movies, num_users> predictedY;
	predictedY =  1, 1, 0, 1,
			 5, 5, 4, 1,
			 5, 5, 5, 1,
			 5, 5, 0, 1,
			 5, 5, 0, 1;
	recommends r = com_firfi::mostRecommend(predictedY,2);
	recommends shouldbe(4);
	map_t map0; map0.insert(pair_t(5,1)); map0.insert(pair_t(5,2));
	shouldbe[0] = map0;
	map_t map1; map1.insert(pair_t(5,1)); map1.insert(pair_t(5,2));
	shouldbe[1] = map1;
	map_t map2; map2.insert(pair_t(5,2)); map2.insert(pair_t(4,1));
	shouldbe[2] = map2;
	map_t map3; map3.insert(pair_t(1,0)); map3.insert(pair_t(1,1));
	shouldbe[3] = map3;
	ASSERT_EQ(r, shouldbe);

}

//TEST(Matrix, AdditionSemantic) {
//	static const long i = 4;
//	static const long j = 5;
//	static const int k = 3;
//	matrix<double,0,1> GDArgs =
//	com_firfi::GDFunc<matrix<double> >::fold(randm(i, k), randm(j, k));
//	matrix<double,0,1> GDArgs2 = GDArgs;
//	double alpha = 5.8562e-65;
//	cout << "good assign start" << endl;
//	GDArgs2 += alpha*GDArgs;
//	cout << "good assign end" << endl;
//	ASSERT_TRUE(GDArgs != GDArgs2);
//}

TEST(RecommendSystem, CollaborativeFiltering) {
	static const long num_users = 4;
	static const long num_movies = 5;
	static const int num_features = 3;
	matrix<double, num_movies, num_users> Y;
	Y =  1, 1, 0, 1, // one of two most preferably movies for user 3
		 5, 5, 4, 1,
		 5, 5, 5, 1,
		 5, 5, 0, 1, // one of two most preferably movies for user 3
		 5, 5, 0, 1; // least preferable movie for user 3
	matrix<double, num_movies, num_users> R;
	R	=	1, 1, 0, 1,
			1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 0, 1,
			1, 1, 0, 1;
	// actually one of ~1k instances of this test fails
	srand(time(NULL));
	matrix<double> rm1 = randm(num_movies, num_features);
	matrix<double> rm2 = randm(num_users, num_features);
	matrix<double,0,1> GDArgs =
			dlib::join_cols(
							dlib::reshape_to_column_vector(rm1),
							dlib::reshape_to_column_vector(rm2)
							);
	matrix<double,0,1> initialGDArgs = GDArgs;
	//double min_delta = 0; int max_iter = 50;
	double lambda = 10;
	com_firfi::GDFunc<matrix<double> > func(Y,R,lambda,num_movies,num_users,num_features);
	com_firfi::GDDer<matrix<double> > der(func);
	dlib::objective_delta_stop_strategy stop_strategy;

	dlib::find_min(dlib::lbfgs_search_strategy(20), stop_strategy, func,
			der, GDArgs, -1);
	matrix<double> X =
	dlib::reshape(dlib::crop_cols(GDArgs, 0, 0 + num_movies*num_features), num_movies, num_features);
	matrix<double> Theta =
	dlib::reshape(dlib::crop_cols(GDArgs, num_movies * num_features, num_movies * num_features + num_users*num_features), num_users, num_features);

	ASSERT_TRUE(GDArgs != initialGDArgs); // GDArgs shouldn't be same at least after optimization
	matrix<double, num_movies, num_users> predictedY = X*dlib::trans(Theta);
	double firstMovie = predictedY(0,2);
	double secondMovie = predictedY(1,2);
	double thirdMovie = predictedY(2,2);
	double fourthMovie = predictedY(3,2);
	double fifthMovie = predictedY(4,2);
	EXPECT_GT(fifthMovie, firstMovie);
	EXPECT_GT(fourthMovie, firstMovie);
	EXPECT_GT(secondMovie, fourthMovie);
	EXPECT_GT(secondMovie, fifthMovie);
	EXPECT_GT(thirdMovie, secondMovie);
	//ASSERT_TRUE(abs(firstMovie-fourthMovie) < 0.1);
	//ASSERT_TRUE(abs(secondMovie-thirdMovie) < 0.1);
}
