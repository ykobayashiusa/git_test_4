#include "libvorticus.h" 
#include "geometry/tetrahedral_quadrules.tpp"
#include "fem/Y_temporal_BC.h"

#include <fstream>
#include <sstream>
#include <iostream>


using namespace fem;

// TODO: define a typedef for element matrices, this is getting quite long.
template <size_t gorder, size_t aorder>
auto elem_mat( const tetrahedron<gorder> &T )
-> arma::mat::fixed< shapefcts3d::num<aorder>(), shapefcts3d::num<aorder>() >
{
    constexpr size_t ndof { shapefcts3d::num<aorder>() }; // Number of shape-functions.
    arma::mat::fixed<ndof,ndof> A( arma::fill::zeros );

    // TODO: Please find out what order actually is necessary.
	const tetrahedral_quadrule rule { geometry::get_tetrahedral_quadrule(4) };
	for ( const tetrahedral_quad_node node : rule )
	{
        shapefcts3d::coeffs<aorder,point> dN  { shapefcts3d::dN<aorder>(node.b) };
		geometry::tensor                Jinv  { T.dChi(node.b).inv() }; 
        real                               w  { node.w*T.vol_elem(node.b)/6 };

        for ( size_t i = 0; i < ndof; ++i )
        for ( size_t j = 0; j < ndof; ++j )
        {
            A(i,j) += w*scal_prod(dN[i]*Jinv,dN[j]*Jinv);
        }
	}	
    return A;
}

// TODO: define a typedef for element vectors, this is getting quite long.
template <size_t gorder, size_t aorder>
auto elem_vec( const tetrahedron<gorder> &T, const grid_function<real,gorder,aorder> &f_grid )
-> arma::vec::fixed< shapefcts3d::num<aorder>() >
{
    constexpr size_t ndof { shapefcts3d::num<aorder>() }; // Number of shape-functions.
    arma::vec::fixed<ndof> f( arma::fill::zeros );

    // TODO: Please find out which order actually is necessary!	
    static tetrahedral_quadrule rule { get_tetrahedral_quadrule(4) };
	for ( tetrahedral_quad_node node : rule )
	{
		shapefcts3d::vals<aorder> N { shapefcts3d::N<aorder>(node.b) };
        real                      w { node.w*T.vol_elem(node.b)/6 };

        for ( size_t i = 0; i < ndof; ++i )
            f(i) += w*N[i]*f_grid.eval(T,node.b);
	}
	return f;
}

template <size_t gorder, size_t aorder>
void assemble_A( math::hash_matrix &A, 
                 const arma::mat::fixed< shapefcts3d::num<aorder>(), shapefcts3d::num<aorder>() > &elemA,   
                 const shapefcts3d::coeffs<aorder,size_t> &node_numbers ) 
{
    constexpr size_t ndof { shapefcts3d::num<aorder>() }; // Number of shape-functions.
    for ( size_t i = 0; i < ndof; ++i )
    for ( size_t j = 0; j < ndof; ++j )
    {
        A( node_numbers[i], node_numbers[j] ) += elemA(i,j);
    }
}

template <size_t gorder, size_t aorder>
void assemble_f( math::vector &f,
                         const arma::vec::fixed< shapefcts3d::num<aorder>() > &elemf,
                         const shapefcts3d::coeffs<aorder,size_t> &node_numbers )
{
    constexpr size_t ndof { shapefcts3d::num<aorder>() }; // Number of shape-functions.
    for ( size_t i = 0; i < ndof; ++i )
        f(node_numbers[i]) += elemf(i);
}


int main(int argc, char *argv[] )
{

	multigrid<1> mg = read_gmsh( argv[1] );
	std::cout<< "Done reading mesh." <<std::endl;

    // Refine the mesh a little bit.
	for ( size_t lvl = 0; lvl < 4; ++lvl )
	{
		// marking mesh, reqired for mesh refinement 
		for ( auto it = mg.grid_begin(lvl); it !=mg.grid_end(lvl); ++it )
		    it->set_ref();
	
		mg.adapt(); // mesh refinement based on multilevel refinement algorithm
    }
    size_t maxlvl = mg.last_level();

	node_numbering<1,1>	numb( mg.grid_begin(maxlvl), mg.grid_end(maxlvl) );
    size_t Nnodes = numb.size();
    size_t Nelems = numb.cell_nodes().size();
	std::cout<< "Number of nodes: " << Nnodes << std::endl;
	std::cout<< "Number of cells: " << Nelems << std::endl;

	grid_function<real,1,1>	f_fct( numb );
	for ( size_t i = 0; i < numb.size(); ++i )
		f_fct( numb(i) ) = 1;


    // Do NOT use a math::matrix for A! A has mostly zero entries.
    math::hash_matrix A( Nnodes, Nnodes );
    math::vector f( Nnodes, arma::fill::zeros );
    for ( auto it = mg.grid_begin( maxlvl ); it != mg.grid_end( maxlvl ); ++it )
    {
        assemble_A<1,1>( A, elem_mat<1,1>(*it)      , numb(*it) );
        assemble_f<1,1>( f, elem_vec<1,1>(*it,f_fct), numb(*it) );
    }

    // This is to demonstrate how few entries are actually non-zero.
    std::cout << "Number of entries in system matrix: " << Nnodes*Nnodes << ".\n";
    std::cout << "Number of non-zero entries: " << A.nnz()  << ".\n";
    std::cout << "Percentage of non-zeros: " << (A.nnz() * 100.0)/(Nnodes*Nnodes) << std::endl;

    math::vector x( Nnodes, arma::fill::zeros );
    // TODO: Implement boundary conditions (BCs).
    // You cannot solve the system without applying BCs!
    temporal_BC ( numb, A, f );
    // math::cg( A, f, x );

	grid_function<real,1,1>	u( numb );
	for ( size_t i = 0; i < numb.size(); ++i )
		u( numb(i) ) = x(i);
    
	// writing out .vtu file for mesh visualization using paraview
	vtkwriter<1> writer( mg.grid_begin(maxlvl), mg.grid_end(maxlvl) );
	writer.register_scalar( u,     "u" );
	writer.register_scalar( f_fct, "f" );

	std::stringstream filename; filename << "poisson.vtu";
	std::ofstream file( filename.str() );
	writer.write( file );

    return EXIT_SUCCESS;
}

