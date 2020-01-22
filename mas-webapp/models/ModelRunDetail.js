'use strict';


module.exports = (sequelize, DataTypes) => {
  const ModelRunDetail = sequelize.define('ModelRunDetail', {

  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'ModelRunDetail'
  });

  ModelRunDetail.associate = function(models) {
    models.ModelRunDetail.hasMany(models.ModelOutput, {
      foreignKey: 'ModelId'
    });
  	models.ModelRunDetail.hasMany(models.IndependentVariableResult, {
      foreignKey: 'ModelId'
    });
    models.ModelRunDetail.hasMany(models.PACFPlots, {
      foreignKey: 'ModelId'
    })
  	models.ModelRunDetail.belongsTo(models.RunDetail, {
      onDelete: "CASCADE",
      foreignKey:'RunId'
    });
    models.ModelRunDetail.hasMany(models.Shortlist, {
      foreignKey:'ModelId'
    });
    models.ModelRunDetail.hasMany(models.BackTestPlots, {
      foreignKey:'ModelId'
    });
  };
  return ModelRunDetail;
};
