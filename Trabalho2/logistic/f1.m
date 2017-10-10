## Copyright (C) 2017 Ignacio Espinoso Ribeiro
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} f1 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Ignacio Espinoso Ribeiro <ra169767@floyd.lab.ic.unicamp.br>
## Created: 2017-10-10

function [f1_score] = f1 (X, theta, y)

  p = sigmoid(X * theta);
  p = (p >= 0.5);
  precision = sum(y & p) /  (sum(y & p) + sum(!y & p));
  recall =   sum(y & p) / (sum(y&p) + sum(y & !p));
  f1_score = 2 * precision * recall / (precision + recall);
endfunction
