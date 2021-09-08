from log import Log
log_obj=Log("Adult Census Income Prediction").log()

from flask import Flask,request,render_template
from flask_cors import cross_origin
from main import IncomePrediction

app=Flask(__name__)

@app.route('/')
@cross_origin()
def home():
    try:
       return render_template('home.html')
    except Exception as e:
        log_obj.error(e)

@app.route('/predict/',methods=['POST','GET'])
@cross_origin()
def main():
    try:
        if request.method=='POST':
            age=request.form['age']
            cg=request.form['cg']
            cl=request.form['cl']
            hpw=request.form['hpw']
            sex=int(request.form['sex'])
            twf=request.form['twf']
            data=[[age,twf,sex,cg,cl,hpw]]
            obj = IncomePrediction()
            result=obj.predict(data)
            if result:
                res="Person income is more than 50K"
            else:
                res="Person income is less than or equal to 50K"

            return render_template('result.html',res=res)

        return render_template('adult.html')
    except Exception as e:
        log_obj.error(e)

@app.route('/visualize/')
@cross_origin()
def visualize():
    try:
        return render_template('profile_report.html')
    except Exception as e:
        return log_obj.error(e)

@app.route('/accuracy/')
@cross_origin()
def accuracy():
    try:
        obj=IncomePrediction()
        mat,result=obj.accuracy()
        print(mat,result)
        return render_template('table.html',mat=mat,result=result)
    except Exception as e:
        return log_obj.error(e)
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)