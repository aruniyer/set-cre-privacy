function answer = PDFBeta(varargin)


% Change it so the output is a cell array with as many solution as is given
% Change ind = find statements in each case






% answer = PDFBeta(t)
%
% answer = PDFBeta(t, A (optional), B (optional), y (optional), a (optional), b (optional))
%
%                                     [A-1]      [B-1]
%              Gamma(A+B)       (t-a)^   . (b-t)^      
%   f(t) = ----------------  .  ----------------------
%                                       [A+B-1]
%          Gamma(A).Gamma(B)      (b-a)^
%   
%  --------------------------------------------------
%  | VAR |       NAME      | DEFAULT |    BOUND     |
%  --------------------------------------------------
%  |  t  | Time  (or randam variable)|   a < t < b  |
%  --------------------------------------------------
%  |  A  | Alpha           |    1    |   0 < A      |
%  --------------------------------------------------
%  |  B  | Beta            |    1    |   0 < B      |
%  --------------------------------------------------
%  |  y  | f(t)            |   '?'   |              |
%  -------------------------------------------------- 
%  |  a  | lower bound     |    0    |   0 < a      |
%  --------------------------------------------------
%  |  b  | upper bound     |    1    |   0 < b      |
%  --------------------------------------------------

%
%   Input variables may be arrays or scaler.
%
%   Place a, '?' in the variable to be solved. If no '?' is present the function
%   will solve for f(t).
%
%   Place an empty input, [] to make variable use default
%
%%%   Eg. To solve for  %%%%%%%%%%%%% TODO  Create an example
%
%   Last Modified by Andrew O'Connor 03-Feb-2009

%Set default values
A = 1;
B = 1;
y = '?';
a = 0;
b = 1;

% Overwrites default values with input arguments
if nargin == 0 || isempty(cell2mat(varargin(1))), h = errordlg('PDFBeta: Requires at least a time input argument.', 'PDFBeta'); answer=[]; return; end;
if nargin >= 1, t = varargin(1); end
if nargin >= 2 & ~isempty(cell2mat(varargin(2))), A = varargin(2); end;
if nargin >= 3 & ~isempty(cell2mat(varargin(3))), B = varargin(3); end;
if nargin >= 4 & ~isempty(cell2mat(varargin(4))), y = varargin(4); end;
if nargin >= 5 & ~isempty(cell2mat(varargin(5))), a = varargin(5); end;
if nargin == 6 & ~isempty(cell2mat(varargin(6))), b = varargin(6); end;
if nargin > 6, h = errordlg('PDFBeta: Too many arguments', 'PDFBeta'); answer=[]; return; end;


%Test to ensure there is only one '?'
question_str = strcmp('?', [t, A, B, y, a, b]);
if sum(question_str)==0, ...    %Array can be created as variables all are cells
        h = errordlg('PDFBeta: No input defined as ?.', 'PDFBeta'); answer=[]; return; end;
if sum(question_str)>1, ...
        h = errordlg('PDFBeta: Too many ? inputs or no y value given to solve for ?.', 'PDFBeta'); answer=[]; return; end;

%Convert all numbers to double or character
if iscell(t), t = cell2mat(t); end
if iscell(A), A = cell2mat(A); end
if iscell(B), B = cell2mat(B); end
if iscell(y), y = cell2mat(y); end
if iscell(a), a = cell2mat(a); end
if iscell(b), b = cell2mat(b); end

% Make all variables the same size
m_size = max([size(t,1), size(A,1), size(B,1), size(y,1), size(a,1), size(b,1)]);
n_size = max([size(t,2), size(A,2), size(B,2), size(y,2), size(a,2), size(b,2)]);

t = repmat(t, m_size/size(t,1), n_size/size(t,2));
A = repmat(A, m_size/size(A,1), n_size/size(A,2));
B = repmat(B, m_size/size(B,1), n_size/size(B,2));
y = repmat(y, m_size/size(y,1), n_size/size(y,2));
a = repmat(a, m_size/size(a,1), n_size/size(a,2));
b = repmat(b, m_size/size(b,1), n_size/size(b,2));

% Validate inputs2
% a less than b
if (sum(a>b)>0) & ~(a(1)=='?'| b(1)=='?'),h = errordlg('PDFBeta: Lower limit a is greater than upper limit b', 'PDFBeta'); answer=[]; return; end
% % Time is between a and b or '?'
% if (sum((t<a & a(1)~='?') | (t>b & b(1)~='?'))>0) & t~='?', h = errordlg('PDFBeta: t value is not between limits a & b', 'PDFBeta'); answer=[]; return; end
% A > 0 or '?'
if (sum(A<0)>0) & A(1)~='?', h = errordlg('PDFBeta: Alpha value is less than zero', 'PDFBeta'); answer=[]; return; end;
% B > 0 or '?'
if (sum(B<0)>0) & B(1)~='?', h = errordlg('PDFBeta: Beta value is less than zero', 'PDFBeta'); answer=[]; return; end;


% Solve Equation Depending on '?' input
switch find(question_str)

% Solve for time    
    case 1  
        guess = (a+b)./2;
        for m = 1:m_size
            for n = 1:n_size
                MyFunc = @(x) ((gamma(A(m,n)+B(m,n))./(gamma(A(m,n)).*gamma(B(m,n)))).*((x-a(m,n)).^(A(m,n)-1).*(b(m,n)-x).^(B(m,n)-1))./((b(m,n)-a(m,n)).^(A(m,n)+B(m,n)-1)))-y(m,n);
                anss = newtzero(MyFunc, guess(m,n));
                ind = find((anss > a(m,n)).*(anss<b(m,n)));
                answer(m,n, 1:length(ind)) = anss(ind);
            end
        end

% Solve for Alpha
    case 2  
        guess = repmat(1, m_size, n_size);
        for m = 1:m_size
            for n = 1:n_size
                MyFunc = @(x) ((gamma(x+B(m,n))./(gamma(x).*gamma(B(m,n)))).*((t(m,n)-a(m,n)).^(x-1).*(b(m,n)-t(m,n)).^(B(m,n)-1))./((b(m,n)-a(m,n)).^(x+B(m,n)-1)))-y(m,n);
                anss = newtzero(MyFunc, guess(m,n));
                ind = find(anss > 0);
                answer(m,n, 1:length(ind)) = anss(ind);
            end
        end
        
        
% Solve for Beta    
    case 3  
        guess = repmat(1, m_size, n_size);
        for m = 1:m_size
            for n = 1:n_size
                MyFunc = @(x) ((gamma(A(m,n)+x)./(gamma(A(m,n)).*gamma(x))).*((t(m,n)-a(m,n)).^(A(m,n)-1).*(b(m,n)-t(m,n)).^(x-1))./((b(m,n)-a(m,n)).^(A(m,n)+x-1)))-y(m,n);
                anss = newtzero(MyFunc, guess(m,n));
                anss = newtzero(MyFunc, guess(m,n));
                ind = find(anss > 0);
                answer(m,n, 1:length(ind)) = anss(ind);
            end
        end
        
% Solve for f(t)
    case 4  
        answer = (gamma(A+B)./(gamma(A).*gamma(B))).*(((t-a).^(A-1).*(b-t).^(B-1))./((b-a).^(A+B-1)));
        mask = (t<a)|(t>b);
        answer(mask)=NaN;

% Solve for lower bound a    
    case 5  
        guess = b-0.5;
        for m = 1:m_size
            for n = 1:n_size
                MyFunc = @(x) ((gamma(A(m,n)+B(m,n))./(gamma(A(m,n)).*gamma(B(m,n)))).*((t(m,n)-x).^(A(m,n)-1).*(b(m,n)-t(m,n)).^(B(m,n)-1))./((b(m,n)-x).^(A(m,n)+B(m,n)-1)))-y(m,n);
                anss = newtzero(MyFunc, guess(m,n));
                ind = find(anss < t(m,n));
                answer(m,n, 1:length(ind)) = anss(ind);
            end
        end
        
        
% Solve for upper bound b
    case 6
        guess = a+0.5;
        for m = 1:m_size
            for n = 1:n_size
                MyFunc = @(x) ((gamma(A(m,n)+B(m,n))./(gamma(A(m,n)).*gamma(B(m,n)))).*((t(m,n)-a(m,n)).^(A(m,n)-1).*(x-t(m,n)).^(B(m,n)-1))./((x-a(m,n)).^(A(m,n)+B(m,n)-1)))-y(m,n);
                anss = newtzero(MyFunc, guess(m,n));
                ind = find(anss > t(m,n));
                answer(m,n, 1:length(ind)) = anss(ind);
            end
        end
end

if ~exist('answer'), answer=[];end

end